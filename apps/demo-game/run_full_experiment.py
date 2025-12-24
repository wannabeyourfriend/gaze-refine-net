import sys
from pathlib import Path
from datetime import datetime
# 把 project_root 加入模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from systematic_drift_calibration_7th_version import run_calibration
from gaze_music_ui_2th import run_music_ui


def main():
    # ========= 基础目录 =========
    BASE_DIR = Path.home() / "Desktop" / "calibration_ui_test"
    BASE_DIR.mkdir(exist_ok=True)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = BASE_DIR / session_ts

    origin_dir = session_dir / "origin"
    music_dir = session_dir / "music"

    origin_dir.mkdir(parents=True)
    music_dir.mkdir(parents=True)

    print(f"Session dir: {session_dir}")

    # ========= Phase 1: 18-point calibration =========
    print("=== Phase 1: calibration ===")
    run_calibration(origin_dir)

    # ========= Phase 2: music UI =========
    print("=== Phase 2: music UI ===")
    run_music_ui(music_dir, origin_dir)


if __name__ == "__main__":
    main()
