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
    """眼动数据流，实时获取并发送gaze坐标 - 优化版"""
    
    # 信号：gaze坐标(x, y, on_surface)
    gaze_signal = pyqtSignal(float, float, bool)
    
    # 信号：连接状态变化
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
        
        # 性能统计
        self.message_count = 0
        self.processing_times = []
        self.last_print_time = time.time()
        
        # 定时器用于定期检查连接
        self.connection_timer = QTimer()
        self.connection_timer.timeout.connect(self._check_connection)
        self.connection_timer.start(2000)  # 每2秒检查一次
        
    def set_window_size(self, width, height):
        """设置窗口尺寸用于坐标转换"""
        self.window_width = width
        self.window_height = height
    
    def start(self):
        """启动眼动追踪"""
        if not HAS_ZMQ:
            print("ZMQ not available. Eye tracker disabled.")
            self.connection_changed.emit(False)
            return False
            
        try:
            # 获取Pupil Capture的PUB端口
            self.ctx = zmq.Context()
            req = self.ctx.socket(zmq.REQ)
            req.connect(f"tcp://127.0.0.1:{self.zmq_port}")
            req.setsockopt(zmq.RCVTIMEO, 2000)  # 2秒超时
            req.send_string("SUB_PORT")
            pub_port = req.recv_string()
            req.close()
            print(f"Connected to Pupil Capture on port: {pub_port}")
            
            # 创建订阅者
            self.sub = self.ctx.socket(zmq.SUB)
            self.sub.connect(f"tcp://127.0.0.1:{pub_port}")
            
            # 优化socket设置
            self.sub.setsockopt(zmq.RCVHWM, 5)      # 只缓存5条消息，丢弃旧数据
            self.sub.setsockopt(zmq.IMMEDIATE, 1)    # 立即发送
            
            # 只订阅surfaces主题（如果你确定需要surface数据）
            self.sub.setsockopt(zmq.SUBSCRIBE, b"surfaces.")
            
            self._running = True
            
            # 启动数据读取定时器 - 更高频率
            self.read_timer = QTimer()
            self.read_timer.timeout.connect(self._read_data_optimized)
            self.read_timer.start(10)  # 每5ms读取一次（200Hz）
            
            self.connection_changed.emit(True)
            print("Eye tracker started successfully.")
            
            return True
            
        except Exception as e:
            print(f"Failed to start eye tracker: {e}")
            self.connection_changed.emit(False)
            self._cleanup()
            return False
    
    def _read_data_optimized(self):
        """优化版：读取眼动数据"""
        if not self._running or not self.sub:
            return
            
        process_start = time.time()
        
        try:
            # 一次性读取所有可用消息
            messages = []
            while True:
                try:
                    topic, payload = self.sub.recv_multipart(flags=zmq.NOBLOCK)
                    messages.append((topic, payload))
                except zmq.Again:
                    break  # 没有更多数据
        
            # 只处理最新的一条消息，丢弃旧的
            if messages:
                # 只取最后一条，避免处理延迟
                topic, payload = messages[-1]
                
                # 快速处理
                if topic.startswith(b"surfaces."):
                    self._process_surface_message(payload)
                    
                self.message_count += 1
                
                # 性能统计
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                
                # 限制统计列表大小
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                    
        except Exception as e:
            # 静默处理异常，不打印以避免影响性能
            pass
    
    def _process_surface_message(self, payload):
        """快速处理surface消息"""
        try:
            # 直接解包，不使用迭代器
            msg = msgpack.unpackb(payload, raw=False)
            
            # 快速提取数据
            gaze_on_surfaces = msg.get("gaze_on_surfaces")
            if gaze_on_surfaces:
                surf = gaze_on_surfaces[0]
                norm_pos = surf.get("norm_pos")
                
                if norm_pos and len(norm_pos) >= 2:
                    sx, sy = norm_pos[0], norm_pos[1]
                    on_surf = surf.get("on_surf", False)
                    
                    # 快速坐标转换（预先计算常量）
                    BORDER = 21
                    width_range = self.window_width - 2 * BORDER
                    height_range = self.window_height - 2 * BORDER
                    
                    x = BORDER + sx * width_range
                    y = BORDER + (1.0 - sy) * height_range
                    
                    # 立即发射信号
                    self.gaze_signal.emit(float(x), float(y), bool(on_surf))
                    
                    self.last_gaze_time = time.time()
                    
        except Exception:
            # 静默处理解析错误
            pass
    
    def _check_connection(self):
        """检查连接状态"""
        if HAS_ZMQ and self._running:
            try:
                # 检查最后接收时间
                current_time = time.time()
                time_since_last = current_time - self.last_gaze_time
                
                if time_since_last > 2.0:  # 2秒没收到数据
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
        """停止眼动追踪"""
        print("Stopping eye tracker...")
        self._running = False
        
        if hasattr(self, 'read_timer'):
            self.read_timer.stop()
        
        self._cleanup()
        self.connection_changed.emit(False)
    
    def _cleanup(self):
        """清理资源"""
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
        """检查是否正在运行"""
        return self._running