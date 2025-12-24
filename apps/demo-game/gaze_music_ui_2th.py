# gaze_music_ui_simple.py
# 简化版UI：直接使用音频分析结果
from datetime import datetime
from pathlib import Path
import sys, time, math
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtCore import QUrl
from eye_tracker_stream import EyeTrackerStream
# 把 project_root 加入模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from post_processing.gaze_calibration_runtime import SimRBFCalibrator

# ---------------- CONFIG ----------------
CIRCLE_RADIUS = 50
APRILTAG_DIR = Path("D:\\UCSD_eyetracking\\pupil_lab_project\\system-drift-calibration\\apriltags")
WINDOW_W, WINDOW_H = 1920, 1080

# 默认基础音符（从C3到C5）
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
        
        # 修正后的统计逻辑
        self.total_red_time = 0.0  # 这个圆圈所有变红时间的总和
        self.total_gaze_time = 0.0  # 注视在这个圆圈内的时间总和
        self.hit_count = 0  # 被激活的次数（每个音1分）
        self.total_score = 0.0  # 这个圆圈的总得分
        
        self.current_red_start = None  # 当前变红开始时间
        self.current_gaze_accumulated = 0.0  # 当前激活的注视时间
        self.SEARCH_DELAY = 0.75  # 秒
    
    def contains(self, x, y):
        return math.hypot(x - self.cx, y - self.cy) <= CIRCLE_RADIUS
    
    def start_red_period(self, start_time):
        """开始一个红色周期"""
        self.current_red_start = start_time
        self.current_gaze_accumulated = 0.0
        self.active = True
    
    def end_red_period(self, end_time):
        """结束一个红色周期并计算得分"""
        if self.current_red_start is not None:
            red_duration = end_time - self.current_red_start
            effective_duration = red_duration - self.SEARCH_DELAY
            if effective_duration > 0:
                score = self.current_gaze_accumulated / effective_duration
            else:
                score = 0.0
            
            # 记录统计
            self.hit_count += 1
            self.total_red_time += red_duration
            self.total_gaze_time += self.current_gaze_accumulated
            self.total_score += score  # 累加得分
            
            # 重置
            self.current_red_start = None
            self.current_gaze_accumulated = 0.0
            self.active = False

    def add_gaze_time(self, gaze_duration, current_time):
        """
        只有在红色开始SEARCH_DELAY秒之后的注视才计分
        """
        if not self.active or self.current_red_start is None:
            return
        elapsed = current_time - self.current_red_start
        # 前 SEARCH_DELAY 秒是 search window，不计分
        if elapsed <= self.SEARCH_DELAY:
            return

        self.current_gaze_accumulated += gaze_duration
    
    def get_score(self):
        """获取这个圆圈的总得分"""
        return self.total_score
    
    def get_average_ratio(self):
        """获取平均注视比例"""
        if self.total_red_time > 0:
            return self.total_gaze_time / self.total_red_time
        return 0.0
    
    def get_average_score_per_hit(self):
        """获取每次激活的平均得分"""
        if self.hit_count > 0:
            return self.total_score / self.hit_count
        return 0.0
    
    def get_stats(self):
        """获取统计信息"""
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
    def __init__(self, audio_file, pitches_file="pitches_result.txt", calibrator=None):
        super().__init__()
        
        # 强制全屏
        self.showFullScreen()
        self.setWindowTitle("Gaze Task")
        self.setStyleSheet("background-color: #F4F1EC;")
        
        # 获取屏幕尺寸
        screen = self.screen().geometry()
        global WINDOW_W, WINDOW_H
        WINDOW_W, WINDOW_H = screen.width(), screen.height()
        
        # 眼动追踪相关
        self.eye_tracker = EyeTrackerStream(WINDOW_W, WINDOW_H)
        self.eye_tracker.gaze_signal.connect(self._on_gaze_received)

        # 上一个圆圈音符
        self.circle_old = None

        # 首先初始化circles列表
        self.circles = []
        self.note_to_circle = {}
        
        # 加载音高序列
        self.pitches = self.load_pitches_file(pitches_file)
        
        # 直接从pitches中提取所有音符
        all_notes_in_audio = set()
        for _, _, note in self.pitches:
            if note:
                all_notes_in_audio.add(note)
        
        # 合并基础音符和音频中的音符
        self.NOTES = list(set(BASE_NOTES) | all_notes_in_audio)
        self.NOTES.sort(key=self.note_to_sort_key)
        
        # 创建圆圈（现在circles已经初始化）
        self.layout_circles_bone()
        
        # AprilTags
        self.apriltags = []  # 确保这个也初始化
        self.load_apriltags()
        
        # 状态变量
        self.start_time = None
        self.current_gaze = None
        self.audio_playing = False
        self.countdown_active = True
        self.countdown_text = ""
        self.song_title_visible = False
        
        # 创建标签
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

        self.calibrator = calibrator

        # 新增：统计相关变量
        self.statistics = {
            'total_score': 0.0,
            'total_red_time': 0.0,
            'total_gaze_time': 0.0,
            'total_hits': 0,
            'notes_played': 0,
            'experiment_start_time': None,
            'experiment_end_time': None
        }
        
        # 当前激活的圆圈跟踪
        self.last_update_time = time.time()
        
        # 添加统计结果显示标签
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
        self.stats_label.hide()  # 初始隐藏，实验结束时显示
        
        
        # 音频播放器
        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.audio_output)
        
        audio_url = QUrl.fromLocalFile(audio_file)
        if audio_url.isValid():
            self.audio_player.setSource(audio_url)
        else:
            print(f"音频文件不可用: {audio_file}")
        
        self.audio_player.playbackStateChanged.connect(self.on_playback_state_changed)
                
        # 启动倒计时
        self.start_countdown()
        # 启动眼动追踪（在显示窗口后）
        QTimer.singleShot(100, self._start_eye_tracker)
        # QTimer.singleShot(100, self._start_simulated_gaze)
        # 更新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_state)
        self.timer.start(10)
        
    def _start_eye_tracker(self):
        """启动眼动追踪"""
        success = self.eye_tracker.start()
        if success:
            print("Eye tracker started successfully")
        else:
            print("Eye tracker failed to start. Using simulated gaze.")
            # 如果眼动追踪失败，回退到模拟模式
            self._start_simulated_gaze()
    
    def _on_gaze_received(self, x, y, on_surface):
        if not on_surface:
            return
        # 原始 gaze
        raw_x, raw_y = x, y
        # 校正 gaze
        if hasattr(self, "calibrator") and self.calibrator:
            corr_x, corr_y = self.calibrator.correct(raw_x, raw_y)
        else:
            corr_x, corr_y = raw_x, raw_y
        self.current_gaze = (int(corr_x), int(corr_y))

    
    def _start_simulated_gaze(self):
        """启动模拟视线（备份方案）"""
        print("Starting simulated gaze...")
        self.simulate_gaze()
        
        # 设置定时器模拟视线移动
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulate_gaze)
        self.sim_timer.start(100)  # 100ms更新一次
        
    def simulate_gaze(self):
        """模拟视线移动（原有的测试函数）"""
        import random
        x = random.randint(100, WINDOW_W-100)
        y = random.randint(100, WINDOW_H-100)
        self.set_gaze(x, y)
    
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        if hasattr(self, 'eye_tracker'):
            self.eye_tracker.stop()
        
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
            
        if self.audio_playing:
            self.audio_player.stop()
        event.accept()

    def note_to_sort_key(self, note: str):
        """
        返回 MIDI 音高（C4 = 60）
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
        """加载音高序列文件"""
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
                print(f"加载文件失败: {e}")
        else:
            print(f"音高文件不存在: {pitches_file}")
        return pitches

    def layout_circles_bone(self):
        self.circles.clear()
        self.note_to_circle.clear()

        notes = sorted(self.NOTES, key=self.note_to_sort_key)
        N = len(notes)

        # ===== 参数 =====
        center_x = WINDOW_W // 2
        base_y = int(WINDOW_H * 0.92)     # 整体靠下
        max_depth = 200                  # 中间凹下去的深度
        horizontal_span = int(WINDOW_W * 0.50)
        min_dx = 110

        BOTTOM_K = 5                     # 最底部多少个点拉平
        RAISE_PIXELS = 300                # 拉平后整体上移多少
        # =====================================

        # ---------- Step 1: 均匀分布 x ----------
        step = max(horizontal_span // (N - 1), min_dx)
        start_x = center_x - step * (N - 1) // 2
        xs = [start_x + i * step for i in range(N)]

        # 归一化到 [-1, 1]
        norm_xs = [(x - center_x) / (horizontal_span / 2) for x in xs]

        # ---------- Step 2: 先算所有点 ----------
        points = []
        for note, x_norm, x in zip(notes, norm_xs, xs):
            y_offset = max_depth * (x_norm ** 2)
            y = base_y - y_offset
            points.append([note, int(x), int(y)])

        # ---------- Step 3: 找最底部 BOTTOM_K 个并拉平 ----------
        # y 越大越靠下
        points_sorted = sorted(points, key=lambda p: p[2], reverse=True)
        bottom_points = points_sorted[:BOTTOM_K]

        # 选一个统一的 y（取它们中较靠上的那个，再整体抬高）
        flat_y = min(p[2] for p in bottom_points) - RAISE_PIXELS

        for p in bottom_points:
            p[2] = flat_y

        BOTTOM_SPREAD_FACTOR = 1.35   # 底排横向放大倍数（1.2–1.5 都很合理）

        # ---------- Step 3.5: 拉开底部 5 个点的 x ----------
        # 先按 x 从左到右排序（保持音高顺序）
        bottom_points_sorted = sorted(bottom_points, key=lambda p: p[1])

        # 当前中心
        bottom_center_x = sum(p[1] for p in bottom_points_sorted) / BOTTOM_K

        for p in bottom_points_sorted:
            dx = p[1] - bottom_center_x
            p[1] = int(bottom_center_x + dx * BOTTOM_SPREAD_FACTOR)

        # ---------- Step 4: 生成圆圈 ----------
        for note, x, y in points:
            c = PianoCircle(x, y, note)
            self.circles.append(c)
            self.note_to_circle[note] = c
    
    def load_apriltags(self):
        """加载AprilTag图片"""
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
            # 输出统计结果
            self.print_final_statistics()
            # 5秒后自动关闭或显示结束界面
            QTimer.singleShot(5000, self.show_final_screen)

    def show_final_screen(self):
        """显示最终结果屏幕"""
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
        
        # 激活当前播放的音符对应的圆圈
        for t0, t1, note in self.pitches:
            if t0 <= t <= t1:
                circle = self.note_to_circle.get(note)
                if circle:
                    # 开始或继续跟踪这个圆圈c
                    if circle.active == False:
                        # 新的激活，开始红色周期
                        circle.start_red_period(current_time)
                        if self.circle_old:
                            self.circle_old.end_red_period(current_time)
                        self.circle_old = circle
                    
                    # 检查当前注视是否在圆圈内
                    if self.current_gaze:
                        if circle.contains(*self.current_gaze):
                            # 添加到圆圈的注视时间
                            circle.add_gaze_time(frame_duration, current_time)
                    
                    # # 对于音符结束的时间段，检查是否需要结束跟踪
                    # if t > t1 - 0.01:  # 接近结束时
                    #     # 结束红色周期并计算得分
                    #     circle.end_red_period(current_time)
        
        self.update()

    def update_overall_statistics(self):
        """更新总体统计数据"""
        total_score = 0.0
        total_red_time = 0.0
        total_gaze_time = 0.0
        total_hits = 0
        max_possible_score = 0  # 满分
        
        for circle in self.circles:
            total_score += circle.total_score
            total_red_time += circle.total_red_time
            total_gaze_time += circle.total_gaze_time
            total_hits += circle.hit_count
            max_possible_score += circle.hit_count  # 每个音最多1分
        
        self.statistics['total_score'] = total_score
        self.statistics['total_red_time'] = total_red_time
        self.statistics['total_gaze_time'] = total_gaze_time
        self.statistics['total_hits'] = total_hits
        self.statistics['max_possible_score'] = max_possible_score
        self.statistics['score_percentage'] = total_score / max_possible_score if max_possible_score > 0 else 0

    def print_final_statistics(self):
        """打印最终统计结果"""
        print("实验结束 - 注视结果统计")
        
        # 重新计算确保数据准确
        self.update_overall_statistics()
        
        print(f"\n📊 基础统计:")
        print(f"   总音数: {self.statistics['total_hits']}")
        print(f"   总变红时间: {self.statistics['total_red_time']:.2f}秒")
        print(f"   总注视时间: {self.statistics['total_gaze_time']:.2f}秒")
        
        if self.statistics['total_red_time'] > 0:
            overall_ratio = self.statistics['total_gaze_time'] / self.statistics['total_red_time']
            print(f"   总体注视比例: {overall_ratio:.1%}")
        
        print(f"\n🎯 得分统计 (核心指标):")
        print(f"   实际得分: {self.statistics['total_score']:.2f} / {self.statistics['max_possible_score']:.0f}")
        print(f"   得分率: {self.statistics['score_percentage']:.1%}")
        
        if self.statistics['total_hits'] > 0:
            avg_score_per_hit = self.statistics['total_score'] / self.statistics['total_hits']
            print(f"   平均每个音得分: {avg_score_per_hit:.3f}")
        
        # 每个音符的统计
        print(f"\n🎵 每个音符的统计 (按总得分排序):")
        notes_with_hits = [c for c in self.circles if c.hit_count > 0]
        if notes_with_hits:
            notes_with_hits.sort(key=lambda c: c.total_score, reverse=True)
            for circle in notes_with_hits:
                stats = circle.get_stats()
                print(f"   {stats['note']:3s}: "
                    f"得分={stats['total_score']:5.2f}, "
                    f"次数={stats['hit_count']:2d}, "
                    f"均分={stats['avg_score_per_hit']:5.3f}, "
                    f"比例={stats['avg_gaze_ratio']:6.1%}")

    def paintEvent(self, e):
        if not hasattr(self, "circles"):
            return
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Arial", 14)
        qp.setFont(font)
        
        # 绘制圆圈
        for c in self.circles:
            # 外圈（鼓边）
            qp.setPen(QColor(120, 90, 60))
            qp.setBrush(QColor(210, 180, 140))
            qp.drawEllipse(
                int(c.cx - CIRCLE_RADIUS - 8),
                int(c.cy - CIRCLE_RADIUS - 8),
                (CIRCLE_RADIUS + 8) * 2,
                (CIRCLE_RADIUS + 8) * 2
            )

            # 内圈（实际 hit 区域）
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

            # 音符文字
            qp.setFont(QFont("Times New Roman", 16))
            qp.setPen(QColor(40, 40, 40))
            qp.drawText(
                int(c.cx - 18),
                int(c.cy + 6),
                c.note
            )

        # 绘制AprilTags
        if len(self.apriltags) >= 4:
            margin = 20
            for i, pixmap in enumerate(self.apriltags[:4]):
                if pixmap:
                    if i == 0: qp.drawPixmap(margin, margin, pixmap)
                    elif i == 1: qp.drawPixmap(WINDOW_W-128-margin, margin, pixmap)
                    elif i == 2: qp.drawPixmap(margin, WINDOW_H-128-margin, pixmap)
                    elif i == 3: qp.drawPixmap(WINDOW_W-128-margin, WINDOW_H-128-margin, pixmap)
        
        # 绘制视线点
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
    log_dir = Path(r"D:\UCSD_eyetracking\pupil_lab_project\system-drift-calibration\judgement_application\logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"

    log_fp = open(log_file, "w", encoding="utf-8")

    sys.stdout = log_fp
    sys.stderr = log_fp

    print(f"Log file: {log_file.resolve()}")
    return log_fp

def run_music_ui(log_dir: Path, origin_dir: Path):
    from datetime import datetime
    origin_csv = origin_dir / "grid_gaze_log.csv"
    calibrator = SimRBFCalibrator(
            origin_csv_path=origin_csv,
            rbf_kernel="multiquadric",
            smooth=1.0
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"music_{timestamp}.log"

    log_fp = open(log_file, "w", encoding="utf-8")
    sys.stdout = log_fp
    sys.stderr = log_fp

    app = QApplication(sys.argv)
    w = GazePianoUI(
        audio_file="Dance_Of_the_Golden_Snake_1.5x.wav",
        pitches_file="Dance_Of_the_Golden_Snake_pitches_1.5x.txt",
        calibrator = calibrator
    )
    exit_code = app.exec()

    log_fp.close()
    sys.exit(exit_code)


if __name__ == '__main__':
    log_fp = redirect_stdout_to_file()
    app = QApplication(sys.argv)
    
    # 使用音高分析结果文件
    audio_file = "Dance_Of_the_Golden_Snake_1.5x.wav"
    pitches_file = "Dance_Of_the_Golden_Snake_pitches_1.5x.txt"
    
    w = GazePianoUI(audio_file, pitches_file)
    
    sys.exit(app.exec())