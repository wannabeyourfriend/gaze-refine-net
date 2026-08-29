# gaze_music_ui_simple.py
# Simplified UI: directly uses audio analysis results
from datetime import datetime
from pathlib import Path
import sys, time, math
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtCore import QUrl
from eye_tracker_stream import EyeTrackerStream

# ---------------- CONFIG ----------------
CIRCLE_RADIUS = 50
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APRILTAG_DIR = PROJECT_ROOT / "model_calibration" / "apriltags"
WINDOW_W, WINDOW_H = 1920, 1080

# Default base notes (C3 to C5)
BASE_NOTES = [
    "C3", "D3", "E3", "F3", "G3",
    "A3", "B3", "C4", "D4", "E4",
    "F4", "G4", "A4", "B4", "C5"
]

# ---------------- UI ELEMENT ----------------
class PianoCircle:
    def __init__(self, cx, cy, note):
        self.cx, self.cy = cx, cy
        self.note = note
        self.active = False
        
        # Statistics tracking
        self.total_red_time = 0.0  # Total time this circle turns red
        self.total_gaze_time = 0.0  # Total gaze time inside this circle
        self.hit_count = 0  # Number of activations (each note counts as 1)
        self.total_score = 0.0  # Aggregate score for this circle
        
        self.current_red_start = None  # Start time of current red period
        self.current_gaze_accumulated = 0.0  # Gaze accumulated during current activation
        self.SEARCH_DELAY = 0.75  # seconds
    
    def contains(self, x, y):
        return math.hypot(x - self.cx, y - self.cy) <= CIRCLE_RADIUS
    
    def start_red_period(self, start_time):
        """Start a red period."""
        self.current_red_start = start_time
        self.current_gaze_accumulated = 0.0
        self.active = True
    
    def end_red_period(self, end_time):
        """End a red period and compute score."""
        if self.current_red_start is not None:
            red_duration = end_time - self.current_red_start
            effective_duration = red_duration - self.SEARCH_DELAY
            if effective_duration > 0:
                score = self.current_gaze_accumulated / effective_duration
            else:
                score = 0.0
            
            # Record statistics
            self.hit_count += 1
            self.total_red_time += red_duration
            self.total_gaze_time += self.current_gaze_accumulated
            self.total_score += score  # Accumulate score
            
            # Reset state
            self.current_red_start = None
            self.current_gaze_accumulated = 0.0
            self.active = False

    def add_gaze_time(self, gaze_duration, current_time):
        """
        Only gaze after SEARCH_DELAY seconds contributes to the score.
        """
        if not self.active or self.current_red_start is None:
            return
        elapsed = current_time - self.current_red_start
        # The initial SEARCH_DELAY seconds are the search window and do not score
        if elapsed <= self.SEARCH_DELAY:
            return

        self.current_gaze_accumulated += gaze_duration
    
    def get_score(self):
        """Get total score for this circle."""
        return self.total_score
    
    def get_average_ratio(self):
        """Get average gaze ratio."""
        if self.total_red_time > 0:
            return self.total_gaze_time / self.total_red_time
        return 0.0
    
    def get_average_score_per_hit(self):
        """Get average score per activation."""
        if self.hit_count > 0:
            return self.total_score / self.hit_count
        return 0.0
    
    def get_stats(self):
        """Return statistics for this circle."""
        return {
            'note': self.note,
            'hit_count': self.hit_count,
            'total_score': self.total_score,
            'avg_score_per_hit': self.get_average_score_per_hit(),
            'avg_gaze_ratio': self.get_average_ratio(),
            'total_red_time': self.total_red_time,
            'total_gaze_time': self.total_gaze_time
        }
    
class CountdownThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def run(self):
        self.update_signal.emit("3")
        time.sleep(1)
        self.update_signal.emit("2")
        time.sleep(1)
        self.update_signal.emit("1")
        time.sleep(1)
        self.update_signal.emit("")
        self.finished_signal.emit()

class GazePianoUI(QWidget):
    def __init__(self, audio_file, pitches_file="pitches_result.txt"):
        super().__init__()
        
        # Force fullscreen
        self.showFullScreen()
        self.setWindowTitle("Gaze Task")
        self.setStyleSheet("background-color: #F4F1EC;")
        
        # Capture screen size
        screen = self.screen().geometry()
        global WINDOW_W, WINDOW_H
        WINDOW_W, WINDOW_H = screen.width(), screen.height()
        
        self.circle_old = None

        # Eye tracking
        self.eye_tracker = EyeTrackerStream(WINDOW_W, WINDOW_H)
        self.eye_tracker.gaze_signal.connect(self._on_gaze_received)

        # Initialize circles
        self.circles = []
        self.note_to_circle = {}
        
        # Load pitch sequence
        self.pitches = self.load_pitches_file(pitches_file)
        
        # Extract all notes from pitch data
        all_notes_in_audio = set()
        for _, _, note in self.pitches:
            if note:
                all_notes_in_audio.add(note)
        
        # Merge base notes and audio notes
        self.NOTES = list(set(BASE_NOTES) | all_notes_in_audio)
        self.NOTES.sort(key=self.note_to_sort_key)
        
        # Create circles
        self.layout_circles_bone()
        
        # AprilTags
        self.apriltags = []  # Ensure initialization
        self.load_apriltags()
        
        # State variables
        self.start_time = None
        self.current_gaze = None
        self.audio_playing = False
        self.countdown_active = True
        self.countdown_text = ""
        self.song_title_visible = False
        
        # Labels
        self.countdown_label = QLabel(self)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet(
            "font-family: 'Times New Roman';"
            "font-size: 72px;"
            "font-weight: bold;"
            "color: #3A3A3A;"
        )
        self.countdown_label.setGeometry(int(WINDOW_W/4), 100, int(WINDOW_W/2), 150)
        
        self.title_label = QLabel("Dance Of the Golden Snake", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            "font-family: 'Times New Roman';"
            "font-size: 72px;"
            "font-weight: bold;"
            "color: #3A3A3A;"
        )
        self.title_label.setGeometry(int(WINDOW_W/4), 100, int(WINDOW_W/2), 150)
        self.title_label.hide()

        # Statistics
        self.statistics = {
            'total_score': 0.0,
            'total_red_time': 0.0,
            'total_gaze_time': 0.0,
            'total_hits': 0,
            'notes_played': 0,
            'experiment_start_time': None,
            'experiment_end_time': None
        }
        
        # Track currently active circle timing
        self.last_update_time = time.time()
        
        # Statistics display label
        self.stats_label = QLabel(self)
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.stats_label.setStyleSheet(
            "font-family: 'Arial';"
            "font-size: 14px;"
            "color: #333333;"
            "background-color: rgba(255, 255, 255, 200);"
            "padding: 8px;"
            "border-radius: 5px;"
        )
        self.stats_label.setGeometry(WINDOW_W - 350, 20, 320, 100)
        self.stats_label.hide()  # Hidden initially; shown at the end
        
        
        # Audio player
        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.audio_output)
        
        audio_url = QUrl.fromLocalFile(audio_file)
        if audio_url.isValid():
            self.audio_player.setSource(audio_url)
        else:
            print(f"Audio file unavailable: {audio_file}")
        
        self.audio_player.playbackStateChanged.connect(self.on_playback_state_changed)
                
        # Start countdown
        self.start_countdown()
        # Start eye tracker after the window is visible
        QTimer.singleShot(100, self._start_eye_tracker)
        # QTimer.singleShot(100, self._start_simulated_gaze)
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_state)
        self.timer.start(10)
        
    def _start_eye_tracker(self):
        """Start eye tracking."""
        success = self.eye_tracker.start()
        if success:
            print("Eye tracker started successfully")
        else:
            print("Eye tracker failed to start. Using simulated gaze.")
            # Fallback to simulated gaze
            self._start_simulated_gaze()
    
    def _on_gaze_received(self, x, y, on_surface):
        """Handle incoming gaze data."""
        # Only update when the point is on the surface
        if on_surface:
            self.current_gaze = (int(x), int(y))
    
    def _start_simulated_gaze(self):
        """Start simulated gaze as a fallback."""
        print("Starting simulated gaze...")
        self.simulate_gaze()
        
        # Timer to move simulated gaze
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulate_gaze)
        self.sim_timer.start(100)  # Update every 100ms
        
    def simulate_gaze(self):
        """Simulate gaze movement (test helper)."""
        import random
        x = random.randint(100, WINDOW_W-100)
        y = random.randint(100, WINDOW_H-100)
        self.set_gaze(x, y)
    
    def closeEvent(self, event):
        """Clean up resources when closing."""
        if hasattr(self, 'eye_tracker'):
            self.eye_tracker.stop()
        
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
            
        if self.audio_playing:
            self.audio_player.stop()
        event.accept()

    def note_to_sort_key(self, note: str):
        """
        Return the MIDI pitch value (C4 = 60).
        """
        NOTE_TO_SEMITONE = {
            "C": 0,  "C#": 1,
            "D": 2,  "D#": 3,
            "E": 4,
            "F": 5,  "F#": 6,
            "G": 7,  "G#": 8,
            "A": 9,  "A#": 10,
            "B": 11,
        }
        if note[-2] == "#":
            name = note[:-2]    # C#
            octave = int(note[-1])
        else:
            name = note[:-1]    # C
            octave = int(note[-1])

        semitone = NOTE_TO_SEMITONE[name]
        midi = 12 * (octave + 1) + semitone
        return midi
    
    def load_pitches_file(self, pitches_file):
        """Load the pitch sequence file."""
        pitches = []
        if Path(pitches_file).exists():
            try:
                with open(pitches_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(',')
                            if len(parts) == 3:
                                pitches.append((float(parts[0]), float(parts[1]), parts[2]))
                print(f"upload {len(pitches)} parts of music from the file.")
            except Exception as e:
                print(f"Failed to load pitch file: {e}")
        else:
            print(f"Pitch file does not exist: {pitches_file}")
        return pitches

    def layout_circles_bone(self):
        self.circles.clear()
        self.note_to_circle.clear()

        notes = sorted(self.NOTES, key=self.note_to_sort_key)
        N = len(notes)

        # Parameters
        center_x = WINDOW_W // 2
        base_y = int(WINDOW_H * 0.92)     # Slightly lower on the screen
        max_depth = 200                  # Depth of the central curve
        horizontal_span = int(WINDOW_W * 0.50)
        min_dx = 110

        BOTTOM_K = 5                     # Number of lowest points to flatten
        RAISE_PIXELS = 300                # Raise flattened points upward
        # =====================================

        # ---------- Step 1: even x distribution ----------
        step = max(horizontal_span // (N - 1), min_dx)
        start_x = center_x - step * (N - 1) // 2
        xs = [start_x + i * step for i in range(N)]

        # Normalize to [-1, 1]
        norm_xs = [(x - center_x) / (horizontal_span / 2) for x in xs]

        # ---------- Step 2: compute base curve ----------
        points = []
        for note, x_norm, x in zip(notes, norm_xs, xs):
            y_offset = max_depth * (x_norm ** 2)
            y = base_y - y_offset
            points.append([note, int(x), int(y)])

        # ---------- Step 3: find the BOTTOM_K lowest points and flatten ----------
        # Larger y means lower on screen
        points_sorted = sorted(points, key=lambda p: p[2], reverse=True)
        bottom_points = points_sorted[:BOTTOM_K]

        # Choose a unified y (take the highest among them and then lift)
        flat_y = min(p[2] for p in bottom_points) - RAISE_PIXELS

        for p in bottom_points:
            p[2] = flat_y

        BOTTOM_SPREAD_FACTOR = 1.35   # Horizontal spread factor for bottom row

        # ---------- Step 3.5: widen x spacing for the bottom 5 points ----------
        # Sort by x from left to right (keep pitch order)
        bottom_points_sorted = sorted(bottom_points, key=lambda p: p[1])

        # Current center
        bottom_center_x = sum(p[1] for p in bottom_points_sorted) / BOTTOM_K

        for p in bottom_points_sorted:
            dx = p[1] - bottom_center_x
            p[1] = int(bottom_center_x + dx * BOTTOM_SPREAD_FACTOR)

        # ---------- Step 4: generate circles ----------
        for note, x, y in points:
            c = PianoCircle(x, y, note)
            self.circles.append(c)
            self.note_to_circle[note] = c
    
    def load_apriltags(self):
        """Load AprilTag images."""
        self.apriltags = []
        for i in range(1, 5):
            possible_names = [
                f"apriltags_tag36h11_0-{i}.jpeg",
            ]
            for filename in possible_names:
                p = APRILTAG_DIR / filename
                if p.exists():
                    pixmap = QPixmap(str(p))
                    if not pixmap.isNull():
                        self.apriltags.append(pixmap.scaled(128, 128))
                        break

    def start_countdown(self):
        self.countdown_thread = CountdownThread()
        self.countdown_thread.update_signal.connect(self.update_countdown)
        self.countdown_thread.finished_signal.connect(self.start_experiment)
        self.countdown_thread.start()
    
    def update_countdown(self, text):
        self.countdown_text = text
        if text:
            self.countdown_label.setText(text)
            self.countdown_label.show()
        else:
            self.countdown_label.hide()
        self.update()
    
    def start_experiment(self):
        print("experiment starts ...")
        self.countdown_active = False
        self.song_title_visible = True
        self.title_label.show()
        
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.statistics['experiment_start_time'] = self.start_time
        
        if self.audio_player.source().isValid():
            self.audio_player.play()
            self.audio_playing = True

    def on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.audio_playing = False
            self.statistics['experiment_end_time'] = time.time()
            # Output statistics
            self.print_final_statistics()
            # Close or show final screen after 5 seconds
            QTimer.singleShot(5000, self.show_final_screen)

    def show_final_screen(self):
        """Display the final results screen."""
        self.update()

    def set_gaze(self, x, y):
        self.current_gaze = (x, y)
    
    def update_state(self):
        if not self.start_time or not self.audio_playing:
            return
        
        current_time = time.time()
        t = current_time - self.start_time
        frame_duration = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Activate the circle for the current note
        for t0, t1, note in self.pitches:
            if t0 <= t <= t1:
                circle = self.note_to_circle.get(note)
                if circle:
                    # Start or continue tracking this circle
                    if circle.active == False:
                        # New activation: start red period
                        circle.start_red_period(current_time)
                        if self.circle_old:
                            self.circle_old.end_red_period(current_time)
                        self.circle_old = circle
                    
                    # Check whether gaze is within the circle
                    if self.current_gaze:
                        if circle.contains(*self.current_gaze):
                            # Accumulate gaze time
                            circle.add_gaze_time(frame_duration, current_time)
        self.update()

    def update_overall_statistics(self):
        """Update aggregate statistics."""
        total_score = 0.0
        total_red_time = 0.0
        total_gaze_time = 0.0
        total_hits = 0
        max_possible_score = 0  # Full score
        
        for circle in self.circles:
            total_score += circle.total_score
            total_red_time += circle.total_red_time
            total_gaze_time += circle.total_gaze_time
            total_hits += circle.hit_count
            max_possible_score += circle.hit_count  # Each activation is worth 1 point
        
        self.statistics['total_score'] = total_score
        self.statistics['total_red_time'] = total_red_time
        self.statistics['total_gaze_time'] = total_gaze_time
        self.statistics['total_hits'] = total_hits
        self.statistics['max_possible_score'] = max_possible_score
        self.statistics['score_percentage'] = total_score / max_possible_score if max_possible_score > 0 else 0

    def print_final_statistics(self):
        """Print final statistics."""
        print("Experiment complete - gaze results summary")
        
        # Recompute to ensure accuracy
        self.update_overall_statistics()
        
        print(f"\n📊 Basic statistics:")
        print(f"   Total notes: {self.statistics['total_hits']}")
        print(f"   Total red time: {self.statistics['total_red_time']:.2f}s")
        print(f"   Total gaze time: {self.statistics['total_gaze_time']:.2f}s")
        
        if self.statistics['total_red_time'] > 0:
            overall_ratio = self.statistics['total_gaze_time'] / self.statistics['total_red_time']
            print(f"   Overall gaze ratio: {overall_ratio:.1%}")
        
        print(f"\n🎯 Score (core metrics):")
        print(f"   Actual score: {self.statistics['total_score']:.2f} / {self.statistics['max_possible_score']:.0f}")
        print(f"   Score rate: {self.statistics['score_percentage']:.1%}")
        
        if self.statistics['total_hits'] > 0:
            avg_score_per_hit = self.statistics['total_score'] / self.statistics['total_hits']
            print(f"   Avg score per note: {avg_score_per_hit:.3f}")
        
        # Per-note statistics
        print(f"\n🎵 Per-note stats (sorted by total score):")
        notes_with_hits = [c for c in self.circles if c.hit_count > 0]
        if notes_with_hits:
            notes_with_hits.sort(key=lambda c: c.total_score, reverse=True)
            for circle in notes_with_hits:
                stats = circle.get_stats()
                print(f"   {stats['note']:3s}: "
                    f"score={stats['total_score']:5.2f}, "
                    f"count={stats['hit_count']:2d}, "
                    f"avg={stats['avg_score_per_hit']:5.3f}, "
                    f"ratio={stats['avg_gaze_ratio']:6.1%}")

    def paintEvent(self, e):
        if not hasattr(self, "circles"):
            return
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Arial", 14)
        qp.setFont(font)
        
        # Draw circles
        for c in self.circles:
            # Outer ring
            qp.setPen(QColor(120, 90, 60))
            qp.setBrush(QColor(210, 180, 140))
            qp.drawEllipse(
                int(c.cx - CIRCLE_RADIUS - 8),
                int(c.cy - CIRCLE_RADIUS - 8),
                (CIRCLE_RADIUS + 8) * 2,
                (CIRCLE_RADIUS + 8) * 2
            )

            # Inner hit area
            if c.active:
                qp.setBrush(QColor(200, 60, 60))
            else:
                qp.setBrush(QColor(235, 220, 200))

            qp.setPen(QColor(60, 60, 60))
            qp.drawEllipse(
                int(c.cx - CIRCLE_RADIUS),
                int(c.cy - CIRCLE_RADIUS),
                CIRCLE_RADIUS * 2,
                CIRCLE_RADIUS * 2
            )

            # Note label
            qp.setFont(QFont("Times New Roman", 16))
            qp.setPen(QColor(40, 40, 40))
            qp.drawText(
                int(c.cx - 18),
                int(c.cy + 6),
                c.note
            )

        # Draw AprilTags
        if len(self.apriltags) >= 4:
            margin = 20
            for i, pixmap in enumerate(self.apriltags[:4]):
                if pixmap:
                    if i == 0: qp.drawPixmap(margin, margin, pixmap)
                    elif i == 1: qp.drawPixmap(WINDOW_W-128-margin, margin, pixmap)
                    elif i == 2: qp.drawPixmap(margin, WINDOW_H-128-margin, pixmap)
                    elif i == 3: qp.drawPixmap(WINDOW_W-128-margin, WINDOW_H-128-margin, pixmap)
        
        # Draw gaze point
        if self.current_gaze:
            qp.setBrush(QColor(0, 0, 255, 180))
            qp.drawEllipse(int(self.current_gaze[0]-10), int(self.current_gaze[1]-10), 20, 20)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        if self.audio_playing:
            self.audio_player.stop()
        event.accept()

def redirect_stdout_to_file():
    log_dir = Path.home() / "Desktop" / "calibration_ui_test" / "origin"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"

    log_fp = open(log_file, "w", encoding="utf-8")

    sys.stdout = log_fp
    sys.stderr = log_fp

    print(f"Log file: {log_file.resolve()}")
    return log_fp

def run_music_ui(log_dir: Path):
    from datetime import datetime

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"music_{timestamp}.log"

    log_fp = open(log_file, "w", encoding="utf-8")
    sys.stdout = log_fp
    sys.stderr = log_fp

    app = QApplication(sys.argv)
    w = GazePianoUI(
        audio_file="Dance_Of_the_Golden_Snake.wav",
        pitches_file="Dance_Of_the_Golden_Snake_pitches.txt"
    )
    exit_code = app.exec()

    log_fp.close()
    sys.exit(exit_code)


if __name__ == '__main__':
    log_fp = redirect_stdout_to_file()
    app = QApplication(sys.argv)
    REPO_ROOT = Path(__file__).resolve().parents[2]
    # Use the analyzed pitch results
    audio_file = REPO_ROOT / "assets" / "demo_files" / "Dance_Of_the_Golden_Snake.wav"
    pitches_file = REPO_ROOT / "assets" / "demo_files" / "Dance_Of_the_Golden_Snake_pitches.txt"
    
    w = GazePianoUI(str(audio_file), str(pitches_file))
    
    sys.exit(app.exec())
