#!/usr/bin/env python3
"""
systematic_drift_calibration.py  （已增强版）

功能要点（按照你的要求）：
- 第一次运行：按原逻辑点亮 6x4 网格、每点等待 2s、采样 4s、剔除距离>250px 样本、保存 samples_target_*.csv 与 grid_gaze_log.csv，并生成两张图（calibration_offsets.png, gaze_clusters.png）。
- 第一次运行结束后：在 session_dir 下新建 "origin" 子文件夹，把第一次产生的 csv/png 文件复制到 origin。
- 自动从第一次的 grid_gaze_log.csv 调用 gaze_correction_rbf.GazeRBFCalibrator 构建 RBF 模型（若模块不存在会给出提示）。
- 等待用户在终端按回车。当用户按回车时开始第二轮运行（RBF 模式）：
  - 第二轮展示与第一次相同的界面与流程，但在每个 target 的 4s 采样结束后，对样本平均位置用 RBF 隐式反解（correct_gaze_iterative）得到校正坐标，并以该校正坐标作为“系统认定的 gaze”保存与后续绘图。
  - 第二轮的所有输出（主 CSV、samples、两个 PNG）保存到 session_dir/RBF/ 下。
- 两轮的输出目录分别为：
    session_dir/origin/       <- 第一次全部文件（备份）
    session_dir/RBF/          <- 第二次所有文件

"""

from __future__ import annotations
import sys
import random
import time
import csv
import math
import shutil
from pathlib import Path
from collections import deque
from datetime import datetime
import pandas as pd

from PyQt6.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal, QObject, QEventLoop
from PyQt6.QtGui import QPainter, QColor, QPixmap, QFont
from PyQt6.QtWidgets import QApplication, QWidget, QLabel


# Optional imports for ZMQ subscriber to Pupil / custom eyetracker
try:
    import zmq
    import msgpack
    HAS_ZMQ = True
except Exception:
    HAS_ZMQ = False

# If gaze_correction_rbf is available we'll import it later when needed.
# from gaze_correction_rbf import GazeRBFCalibrator  # imported dynamically when required

# ----------------- Config -----------------
GRID_COLS = 4  
GRID_ROWS = 3
DOT_RADIUS = 10
HIGHLIGHT_RADIUS = 10
INITIAL_WAIT_MS = 8000  # 启动等待时间
APRILTAG_DIR = Path(__file__).resolve().parent / "apriltags"
ZMQ_REQ_PORT = 50020

# sampling params
PRE_WAIT_MS = 2000   # 亮起后等待 2s
SAMPLE_DURATION_MS = 4000  # 采样 4s
MAX_SAMPLE_DIST_PX = 500  # 剔除半径

# ------------------------------------------
class EyeTrackerWorker(QObject):
    """ZMQ worker same as before"""
    gaze = pyqtSignal(int, int, bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.window_width = 1000
        self.window_height = 1000
        self.ctx = None
        self.sub = None
        self.unpacker = None
        self.world_topic = b"gaze."
        self.surf_topic = b"surfaces."

    def set_window_size(self, w: int, h: int):
        self.window_width = w
        self.window_height = h

    def start(self):
        if not HAS_ZMQ:
            print("ZMQ not installed; EyeTrackerWorker will be disabled.")
            return

        self._running = True
        self.ctx = zmq.Context()
        try:
            req = self.ctx.socket(zmq.REQ)
            req.connect(f"tcp://127.0.0.1:{ZMQ_REQ_PORT}")
            req.send_string("SUB_PORT")
            pub_port = req.recv_string()
            print(f"Pupil PUB port: {pub_port}")

            self.sub = self.ctx.socket(zmq.SUB)
            self.sub.connect(f"tcp://127.0.0.1:{pub_port}")
            for t in (self.world_topic, self.surf_topic):
                self.sub.setsockopt(zmq.SUBSCRIBE, t)
            self.unpacker = msgpack.Unpacker(raw=False)

            # ✅ 改这里：启动一个 QTimer，周期调用 run_loop 一次
            self.timer = QTimer()
            self.timer.timeout.connect(self.run_once)
            self.timer.start(5)  # 每5ms 调用一次

        except Exception as e:
            print(f"EyeTrackerWorker failed to start ZMQ: {e}")
            self._running = False

    def run_once(self):
        """非阻塞采集一次 gaze 数据"""
        if not self._running or self.sub is None:
            return
        BORDER = 21
        try:
            topic, payload = self.sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return  # 没数据就跳过
        except zmq.ZMQError as e:
            print("ZMQ socket error:", e)
            self._running = False
            return

        self.unpacker.feed(payload)
        for msg in self.unpacker:
            if not self._running:
                break

            # 只处理 surfaces 数据
            if topic.startswith(self.surf_topic):
                gos = msg.get("gaze_on_surfaces") or []
                if not gos:
                    continue

                surf = gos[0]
                sx, sy = surf.get("norm_pos", (None, None))
                on_surf = bool(surf.get("on_surf", False))
                if sx is None or sy is None:
                    continue

                x = int(BORDER + sx * (self.window_width - 2 * BORDER))
                y = int(BORDER + (1.0 - sy) * (self.window_height - 2 * BORDER))
                self.gaze.emit(x, y, on_surf)

    def stop(self):
        """安全停止线程"""
        print("🛑 Stopping EyeTrackerWorker...")
        self._running = False
        if self.timer:
            self.timer.stop()
            self.timer = None
        try:
            if self.sub:
                self.sub.close()
                self.sub = None
            if self.ctx:
                self.ctx.term()
                self.ctx = None
        except Exception as e:
            print("⚠️ Cleanup error:", e)
        print("👁️ EyeTrackerWorker stopped cleanly.")


class GazeOverlay(QWidget):
    """Transparent overlay draws gaze dot"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.last_point = None
        self.radius = 10
        self.show()

    def set_point(self, x: int, y: int):
        self.last_point = QPoint(int(x), int(y))
        self.update()

    def clear(self):
        self.last_point = None
        self.update()

    def paintEvent(self, e):
        if not self.last_point:
            return
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setBrush(QColor(0, 0, 255))
        qp.setPen(Qt.PenStyle.NoPen)
        qp.drawEllipse(self.last_point, self.radius, self.radius)


class DotLabel(QLabel):
    def __init__(self, center_x: int, center_y: int, radius: int, parent=None):
        super().__init__(parent)
        self.cx = center_x
        self.cy = center_y
        self.r = radius
        self.tested = False
        self.lit = False
        self.setFixedSize(radius * 2 + 10, radius * 2 + 10)
        self.move(int(center_x - self.r), int(center_y - self.r))
        self.show()

    def paintEvent(self, e):
        # if not self.lit:
        #     return  # 不绘制黑色点
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.lit:
            qp.setBrush(QColor(255, 0, 0))
            draw_r = HIGHLIGHT_RADIUS
        else:
            qp.setBrush(QColor(0, 0, 0))
            draw_r = self.r
        qp.setPen(Qt.PenStyle.NoPen)
        center = self.rect().center()
        qp.drawEllipse(center, draw_r, draw_r)

    def set_lit(self, on: bool):
        self.lit = on
        self.update()


class GridTester(QWidget):
    """
    GridTester: supports two modes:
      - origin mode (default): original collection & save into LOG_DIR
      - rbf mode: uses an external calibrator to correct mean gaze before logging
    When creating an instance, pass:
      - log_dir: Path to save outputs for this run
      - rbf_calibrator: None for origin run; GazeRBFCalibrator instance for rbf run
    """
    closed = pyqtSignal()  # ✅ 新增信号
    def __init__(self, log_dir: Path, rbf_calibrator=None, poly_calibration = None, custom_points=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pupil Grid Tester")
        self.showFullScreen()
        self.setStyleSheet("background: white;")
        self.w = self.frameGeometry().width()
        self.h = self.frameGeometry().height()

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = self.log_dir / "grid_gaze_log.csv"
        self.rbf_calibrator = rbf_calibrator
        self.rbf_mode = (rbf_calibrator is not None)
        self.poly_calibration = poly_calibration
        self.poly_mode = (poly_calibration is not None)
        self.custom_points = custom_points
        
        # AprilTag images in corners if available
        self.corner_imgs = []
        for i in range(1, 5):
            p = APRILTAG_DIR / f"apriltags_tag36h11_0-{i}.jpeg"
            lab = QLabel(self)
            if p.exists():
                pix = QPixmap(str(p)).scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
                lab.setPixmap(pix)
            else:
                lab.setText("")
            lab.setFixedSize(128, 128)
            lab.setStyleSheet("background: transparent;")
            lab.raise_()
            self.corner_imgs.append(lab)

        # compute grid centers (evenly distributed)
        margin = -100
        left = margin
        right = self.w - margin
        top = margin
        bottom = self.h - margin

        cols = GRID_COLS
        rows = GRID_ROWS
        xs = [left + (right - left) * (i + 0.5) / cols for i in range(cols)]
        ys = [top + (bottom - top) * (j + 0.5) / rows for j in range(rows)]

        # create dot widgets
        self.dots = []
        if custom_points is not None:
            for (cx, cy) in custom_points:
                d = DotLabel(cx, cy, DOT_RADIUS, parent=self)
                self.dots.append(d)
        else:
            for j in range(rows):
                for i in range(cols):
                    cx = xs[i]
                    cy = ys[j]
                    d = DotLabel(cx, cy, DOT_RADIUS, parent=self)
                    self.dots.append(d)
            self.add_interstitial_points()

        # place apriltags
        s = 128
        self.corner_imgs[0].move(0, 0); self.corner_imgs[0].show()
        self.corner_imgs[1].move(self.w - s, 0); self.corner_imgs[1].show()
        self.corner_imgs[2].move(0, self.h - s); self.corner_imgs[2].show()
        self.corner_imgs[3].move(self.w - s, self.h - s); self.corner_imgs[3].show()

        # gaze overlay
        self.gaze_overlay = GazeOverlay(self)
        self.gaze_overlay.resize(self.size())
        self.gaze_overlay.raise_()

        # Eye tracker worker
        self.tracker_thread = None
        self.tracker_worker = None
        self._setup_tracker()

        # logging CSV header
        with self.log_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # keep participant in header for traceability
            w.writerow(["timestamp","target_index","target_x","target_y","original_gaze_x","original_gaze_y","original_dx","original_dy","spread","n_samples"])

        # experiment state
        self.untested_indices = list(range(len(self.dots)))
        random.shuffle(self.untested_indices)
        self.current_target = None

        # gaze buffer for sampling
        self.gaze_buffer = deque()
        self.middle_buffer = deque()

        # start after initial wait
        QTimer.singleShot(INITIAL_WAIT_MS, self._start_next_trial)

        # resize timer
        self.resize_timer = QTimer(self)
        self.resize_timer.setInterval(200)
        self.resize_timer.timeout.connect(self._ensure_overlay_size)
        self.resize_timer.start()

    def _ensure_overlay_size(self):
        if self.gaze_overlay.size() != self.size():
            self.gaze_overlay.resize(self.size())

    def _setup_tracker(self):
        if HAS_ZMQ:
            self.tracker_worker = EyeTrackerWorker()
            self.tracker_worker.set_window_size(self.w, self.h)
            self.tracker_thread = QThread()
            self.tracker_worker.moveToThread(self.tracker_thread)
            self.tracker_thread.started.connect(self.tracker_worker.start)
            self.tracker_worker.gaze.connect(self._on_gaze_from_tracker)
            self.tracker_thread.start()
        else:
            print("ZMQ not available — using mouse position as gaze source")
            self.setMouseTracking(True)

    def mouseMoveEvent(self, ev):
        if not HAS_ZMQ:
            x = ev.position().x()
            y = ev.position().y()
            self._on_gaze_point(int(x), int(y), True)

    def _on_gaze_from_tracker(self, x: int, y: int, on_surf: bool):
        self._on_gaze_point(x, y, on_surf)
        self.havent_on_gaze_point(x, y, on_surf)

    def _on_gaze_point(self, x, y, on_surf):
        self.gaze_overlay.set_point(x, y)
        if not getattr(self, "sampling", False):
            return
        ts = time.time()
        self.gaze_buffer.append((ts, x, y))

    def havent_on_gaze_point(self, x, y, on_surf):
        if not getattr(self, "unsampling", False):
            return
        ts = time.time()
        self.middle_buffer.append((ts, x, y))

    def _start_next_trial(self):
        if not self.untested_indices:
            print("All targets tested — ending this run")
            # QTimer.singleShot(500, self._show_summary_plot)
            return

        self.current_target = random.choice(self.untested_indices)

        tgt = self.dots[self.current_target]
        tgt.set_lit(True)
        print(f"Target {self.current_target} ON")

        self.gaze_buffer.clear()
        self.middle_buffer.clear()
        self.sampling = False
        self.unsampling = False
        QTimer.singleShot(0, self.havent_start_gaze_recording)
        QTimer.singleShot(PRE_WAIT_MS, self._start_gaze_recording)
        QTimer.singleShot(PRE_WAIT_MS + SAMPLE_DURATION_MS, self._commit_current_target)

    def _start_gaze_recording(self):
        self.unsampling = False
        print(f"Start gaze recording for target {self.current_target}")
        self.sampling = True
        self.gaze_buffer.clear()

    def havent_start_gaze_recording(self):
        self.unsampling = True
        self.middle_buffer.clear()

    def _commit_current_target(self):
        if self.current_target is None:
            print("⚠️ No current target to commit")
            return
        self.sampling = False
        print(f"Stop gaze recording for target {self.current_target}")

        tm = [p[0] for p in self.middle_buffer]
        xm = [p[1] for p in self.middle_buffer]
        ym = [p[2] for p in self.middle_buffer]

        unfiltered = [(t, x, y) for t, x, y in zip(tm, xm, ym)]
                      
        ts = [p[0] for p in self.gaze_buffer]
        xs = [p[1] for p in self.gaze_buffer]
        ys = [p[2] for p in self.gaze_buffer]

        if not xs:
            print("⚠️ No gaze samples captured")
            return

        tgt = self.dots[self.current_target]
        tx, ty = tgt.cx, tgt.cy

        # filter by distance
        filtered = [(t, x, y) for t, x, y in zip(ts, xs, ys)
                    if ((x - tx)**2 + (y - ty)**2)**0.5 <= MAX_SAMPLE_DIST_PX]
        if not filtered:
            print("⚠️ No valid samples within radius")
            return

        _, fx, fy = zip(*filtered)
        mean_x = sum(fx) / len(fx)
        mean_y = sum(fy) / len(fy)

        spread = math.sqrt(sum((x - mean_x)**2 + (y - mean_y)**2 for _, x, y in filtered) / len(filtered))
        if spread > 100 and self.custom_points is None:
            # cleanup and next
            self.dots[self.current_target].tested = True
            self.dots[self.current_target].set_lit(False)
            self.current_target = None
            QTimer.singleShot(400, self._start_next_trial)
            print(f'spread is {spread}. check again.')
            return
        else:
            self.untested_indices.remove(self.current_target)
            print(f'spread is {spread}. checked.')
        original_dx, original_dy = mean_x - tx, mean_y - ty

        original_gaze_x = mean_x
        original_gaze_y = mean_y
        
        # save unsamples file
        unsamples_path = self.log_dir / f"samples_target_{self.current_target}_before.csv"
        with unsamples_path.open("w", newline="", encoding="utf-8") as fs:
            ws = csv.writer(fs)
            ws.writerow(["x", "y"])
            for t, x, y in unfiltered:
                ws.writerow([t, int(x), int(y)])
        print(f"✅ Saved {unsamples_path.name} with {len(unfiltered)} samples")

        # save samples file
        samples_path = self.log_dir / f"samples_target_{self.current_target}.csv"
        with samples_path.open("w", newline="", encoding="utf-8") as fs:
            ws = csv.writer(fs)
            ws.writerow(["x", "y"])
            for t, x, y in filtered:
                ws.writerow([t, int(x), int(y)])
        print(f"✅ Saved {samples_path.name} with {len(filtered)} samples")

        # write to main CSV
        t = datetime.now().isoformat(timespec="milliseconds")
        with self.log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([t, self.current_target, int(tx), int(ty), original_gaze_x, original_gaze_y,
                        int(original_dx), int(original_dy), round(spread, 2), len(filtered)])

        # cleanup and next
        self.dots[self.current_target].tested = True
        self.dots[self.current_target].set_lit(False)
        self.current_target = None
        QTimer.singleShot(400, self._start_next_trial)

    def add_interstitial_points(self):
        margin = -100
        left = margin
        right = self.w - margin
        top = margin
        bottom = self.h - margin

        cols = GRID_COLS
        rows = GRID_ROWS

        # 原网格中心坐标
        xs = [left + (right - left) * (i + 0.5) / cols for i in range(cols)]
        ys = [top + (bottom - top) * (j + 0.5) / rows for j in range(rows)]

        # 新增 (rows-1)*(cols-1) 个中点
        new_points = []
        for j in range(rows - 1):
            for i in range(cols - 1):
                cx = (xs[i] + xs[i + 1]) / 2
                cy = (ys[j] + ys[j + 1]) / 2
                new_points.append((cx, cy))

        # 生成 DotLabel 对象并加入 dots 列表
        for (cx, cy) in new_points:
            d = DotLabel(cx, cy, DOT_RADIUS, parent=self)
            self.dots.append(d)

    def closeEvent(self, ev):
        print("🧹 Closing GridTester, stopping tracker thread...")
        if self.tracker_worker:
            self.tracker_worker.stop()
        if self.tracker_thread:
            self.tracker_thread.quit()  # ✅ 现在事件循环是活的，可以用 quit()
            self.tracker_thread.wait()
        self.closed.emit()
        super().closeEvent(ev)


# -----------------------
# Main entrypoint
# -----------------------
def main():
    # -------- 被试目录初始化 --------
    BASE_SAVE_DIR = Path(r"C:\\Users\\Liu Jiaqi\\Desktop\\systematic_recalibration")

    # 询问被试名称
    participant_name = input("Please input the participant name: ").strip()
    if not participant_name:
        participant_name = "Unnamed"

    # 建立被试主文件夹
    participant_dir = BASE_SAVE_DIR / participant_name
    participant_dir.mkdir(parents=True, exist_ok=True)

    # 建立时间戳子文件夹
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = participant_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    # 设置全局保存路径变量（主 CSV）
    LOG_DIR = session_dir

    print(f"\n✅ data will be saved in: {LOG_DIR}\n")
    app = QApplication(sys.argv)
    # Phase 1
    origin_dir = session_dir / "origin"
    origin_dir.mkdir(parents=True, exist_ok=True)
    gt1 = GridTester(log_dir=origin_dir, rbf_calibrator=None)

    gt1.show()
    # run Qt app until user closes the window or it completes
    app.exec()
    del app

    # After first run completes (app.exec returned)
    print("Phase 1 finished. Creating origin backup and building RBF model...")

    # Wait for user to press Enter to start second run
    input("Press Enter to start the second run ...")

    # Start a fresh QApplication and second GridTester
    # Note: some platforms don't allow multiple app.exec cycles; to be safe, create new widgets and exec again.
    # 读取 origin 的目标坐标
    df = pd.read_csv(origin_dir / "grid_gaze_log.csv")
    orig_points = list(zip(df["target_x"], df["target_y"]))
    # 随机选取 4 个旧点
    old_selected = random.sample(orig_points, 4)

    def too_close(p1, p2, min_dist=50):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < min_dist

    w, h = 1920, 1080
    min_dist = 50
    target_num = 28

    new_points = []
    while len(new_points) < target_num:
        p = (random.randint(100, w-100), random.randint(100, h-100))
        if any(too_close(p, op, min_dist) for op in old_selected):
            continue
        if any(too_close(p, np, min_dist) for np in new_points):
            continue
        new_points.append(p)

    # 合并打乱
    test_points = old_selected + new_points
    random.shuffle(test_points)

    app2 = QApplication(sys.argv)
    test1_dir = session_dir / "test"
    gt2 = GridTester(log_dir=test1_dir, custom_points=test_points)
    gt2.show()
    app2.exec()
    print("Phase 2 finished.")

def run_calibration(session_origin_dir: Path):
    app = QApplication(sys.argv)
    gt = GridTester(log_dir=session_origin_dir)
    gt.show()
    app.exec()

if __name__ == "__main__":
    main()
