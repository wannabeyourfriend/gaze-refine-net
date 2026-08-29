import sys
from pathlib import Path
from datetime import datetime
# Add project_root to the module search path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from apps.model_calibration.systematic_drift_calibration import run_calibration
from apps.demo_game.run_demo_game import run_music_ui_refined, run_music_ui_poly, run_music_ui_simrbf


def main():
    # ========= Base Directory =========
    BASE_DIR = Path.home() / "Desktop" / "calibration_ui_test"
    BASE_DIR.mkdir(exist_ok=True)

    # Ask participant to choose UI mode
    print("Please select the music UI mode:")
    print("1. Poly mode")
    print("2. SimRBF mode")
    print("3. SimRBF with neural refined")
    
    while True:
        try:
            choice = int(input("Please enter your choice (1 or 2 or 3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Please enter a valid option (1 or 2 or 3)")
        except ValueError:
            print("Please enter a number (1 or 2 or 3)")
    
    # Set up directories based on selection
    if choice == 1:
        ui_mode = "poly"
        session_base_dir = BASE_DIR / "poly"
        ui_function = run_music_ui_poly
    elif choice == 2:
        ui_mode = "simrbf"
        session_base_dir = BASE_DIR / "simrbf"
        ui_function = run_music_ui_simrbf
    elif choice == 3:
        ui_mode = "simrbf_refined"
        session_base_dir = BASE_DIR / "simrbf_refined"
        ui_function = run_music_ui_refined
    
    session_base_dir.mkdir(exist_ok=True)

    # Create timestamped session directory under the mode directory
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = session_base_dir / session_ts

    origin_dir = session_dir / "origin"
    music_dir = session_dir / "music"

    origin_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected UI mode: {ui_mode}")
    print(f"Session directory: {session_dir}")

    # ========= Phase 1: 18-point calibration =========
    print("=== Phase 1: Calibration ===")
    run_calibration(origin_dir)

    # ========= Phase 2: Music UI =========
    print(f"=== Phase 2: Music UI ({ui_mode}) ===")
    ui_function(music_dir, origin_dir)


if __name__ == "__main__":
    main()
