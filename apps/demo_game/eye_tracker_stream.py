# eye_tracker_stream.py
import sys
import time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

try:
    import zmq
    import msgpack
    HAS_ZMQ = True
except Exception as e:
    print(f"ZMQ import failed: {e}")
    HAS_ZMQ = False

class EyeTrackerStream(QObject):
    """Gaze data stream that emits gaze coordinates in real time (optimized version)."""
    
    # Signal: gaze coordinates (x, y, on_surface)
    gaze_signal = pyqtSignal(float, float, bool)
    
    # Signal: connection status changes
    connection_changed = pyqtSignal(bool)
    
    def __init__(self, window_width=1920, window_height=1080, zmq_port=50020, parent=None):
        super().__init__(parent)
        self.window_width = window_width
        self.window_height = window_height
        self.zmq_port = zmq_port
        
        self._running = False
        self.ctx = None
        self.sub = None
        self.last_gaze_time = time.time()
        
        # Performance stats
        self.message_count = 0
        self.processing_times = []
        self.last_print_time = time.time()
        
        # Timer to periodically check the connection
        self.connection_timer = QTimer()
        self.connection_timer.timeout.connect(self._check_connection)
        self.connection_timer.start(2000)  # Check every 2 seconds
        
    def set_window_size(self, width, height):
        """Set window size for coordinate conversion."""
        self.window_width = width
        self.window_height = height
    
    def start(self):
        """Start the eye tracker."""
        if not HAS_ZMQ:
            print("ZMQ not available. Eye tracker disabled.")
            self.connection_changed.emit(False)
            return False
            
        try:
            # Fetch the Pupil Capture PUB port
            self.ctx = zmq.Context()
            req = self.ctx.socket(zmq.REQ)
            req.connect(f"tcp://127.0.0.1:{self.zmq_port}")
            req.setsockopt(zmq.RCVTIMEO, 2000)  # 2-second timeout
            req.send_string("SUB_PORT")
            pub_port = req.recv_string()
            req.close()
            print(f"Connected to Pupil Capture on port: {pub_port}")
            
            # Create subscriber
            self.sub = self.ctx.socket(zmq.SUB)
            self.sub.connect(f"tcp://127.0.0.1:{pub_port}")
            
            # Optimize socket settings
            self.sub.setsockopt(zmq.RCVHWM, 5)      # Keep only the latest 5 messages
            self.sub.setsockopt(zmq.IMMEDIATE, 1)    # Send immediately
            
            # Subscribe to surfaces topic (when surface data is needed)
            self.sub.setsockopt(zmq.SUBSCRIBE, b"surfaces.")
            
            self._running = True
            
            # Start timer to read data at high frequency
            self.read_timer = QTimer()
            self.read_timer.timeout.connect(self._read_data_optimized)
            self.read_timer.start(10)  # Read every 10ms (100Hz)
            
            self.connection_changed.emit(True)
            print("Eye tracker started successfully.")
            
            return True
            
        except Exception as e:
            print(f"Failed to start eye tracker: {e}")
            self.connection_changed.emit(False)
            self._cleanup()
            return False
    
    def _read_data_optimized(self):
        """Optimized gaze data reader."""
        if not self._running or not self.sub:
            return
            
        process_start = time.time()
        
        try:
            # Read all available messages at once
            messages = []
            while True:
                try:
                    topic, payload = self.sub.recv_multipart(flags=zmq.NOBLOCK)
                    messages.append((topic, payload))
                except zmq.Again:
                    break  # No more data
        
            # Only process the latest message and drop older ones
            if messages:
                # Take the latest to reduce latency
                topic, payload = messages[-1]
                
                # Fast path
                if topic.startswith(b"surfaces."):
                    self._process_surface_message(payload)
                    
                self.message_count += 1
                
                # Collect timing
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                
                # Keep stats list bounded
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                    
        except Exception as e:
            # Swallow unexpected errors to avoid console spam
            pass
    
    def _process_surface_message(self, payload):
        """Process a surface message quickly."""
        try:
            # Unpack payload directly
            msg = msgpack.unpackb(payload, raw=False)
            
            # Extract data quickly
            gaze_on_surfaces = msg.get("gaze_on_surfaces")
            if gaze_on_surfaces:
                surf = gaze_on_surfaces[0]
                norm_pos = surf.get("norm_pos")
                
                if norm_pos and len(norm_pos) >= 2:
                    sx, sy = norm_pos[0], norm_pos[1]
                    on_surf = surf.get("on_surf", False)
                    
                    # Fast coordinate conversion
                    BORDER = 21
                    width_range = self.window_width - 2 * BORDER
                    height_range = self.window_height - 2 * BORDER
                    
                    x = BORDER + sx * width_range
                    y = BORDER + (1.0 - sy) * height_range
                    
                    # Emit immediately
                    self.gaze_signal.emit(float(x), float(y), bool(on_surf))
                    
                    self.last_gaze_time = time.time()
                    
        except Exception:
            # Ignore parse errors silently
            pass
    
    def _check_connection(self):
        """Check connection status."""
        if HAS_ZMQ and self._running:
            try:
                # Check the time since the last message
                current_time = time.time()
                time_since_last = current_time - self.last_gaze_time
                
                if time_since_last > 2.0:  # No data for 2 seconds
                    print("Eye tracker connection timeout.")
                    self._running = False
                    self.connection_changed.emit(False)
                else:
                    self.connection_changed.emit(True)
                    
            except Exception as e:
                print(f"Connection check error: {e}")
                self._running = False
                self.connection_changed.emit(False)
    
    def stop(self):
        """Stop the eye tracker."""
        print("Stopping eye tracker...")
        self._running = False
        
        if hasattr(self, 'read_timer'):
            self.read_timer.stop()
        
        self._cleanup()
        self.connection_changed.emit(False)
    
    def _cleanup(self):
        """Release sockets and ZMQ context."""
        try:
            if self.sub:
                self.sub.close()
                self.sub = None
            if self.ctx:
                self.ctx.term()
                self.ctx = None
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def is_running(self):
        """Check whether the tracker is running."""
        return self._running
