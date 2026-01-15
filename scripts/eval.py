import sys
import csv
import math
from typing import List, Tuple


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def parse_csv_file(filepath: str, method: str) -> Tuple[List[float], int, int]:
    distances, total_count, valid_count = [], 0, 0

    column_map = {
        "original": ("original_gaze_x", "original_gaze_y"),
        "sim_rbf": ("sim_rbf_gaze_x", "sim_rbf_gaze_y"),
        "pred_gaze": ("pred_gaze_x", "pred_gaze_y"),
    }

    if method not in column_map:
        raise ValueError(f"Unknown method: {method}. Must be 'original', 'sim_rbf', or 'pred_gaze'")

    gaze_x_col, gaze_y_col = column_map[method]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty or invalid")

            required_cols = ['target_x', 'target_y', gaze_x_col, gaze_y_col]
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns for method '{method}': {missing_cols}\n"
                    f"Available columns: {reader.fieldnames}"
                )

            for row in reader:
                total_count += 1
                try:
                    target_x = float(row['target_x'])
                    target_y = float(row['target_y'])
                    gaze_x = float(row[gaze_x_col])
                    gaze_y = float(row[gaze_y_col])
                    distance = calculate_distance(target_x, target_y, gaze_x, gaze_y)
                    distances.append(distance)
                    valid_count += 1
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row {total_count} due to error: {e}", file=sys.stderr)

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    return distances, valid_count, total_count


def calculate_statistics(distances: List[float]) -> dict:
    if not distances:
        return {}

    avg_distance = sum(distances) / len(distances)
    variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)

    return {
        'average': avg_distance,
        'min': min(distances),
        'max': max(distances),
        'std_dev': math.sqrt(variance)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_gaze.py <method> [data_file]")
        print("Methods: original, sim_rbf, pred_gaze")
        print("\nExample: python evaluate_gaze.py original data.csv")
        sys.exit(1)

    method = sys.argv[1].lower()
    data_file = sys.argv[2] if len(sys.argv) > 2 else "data.csv"

    valid_methods = ['original', 'sim_rbf', 'pred_gaze']
    if method not in valid_methods:
        print(f"Error: Invalid method '{method}'")
        print(f"Valid methods: {', '.join(valid_methods)}")
        sys.exit(1)

    try:
        print(f"Evaluating method: {method}")
        print(f"Reading data from: {data_file}")
        print("-" * 60)

        distances, valid_count, total_count = parse_csv_file(data_file, method)

        if not distances:
            print("Error: No valid data points found in the file.")
            sys.exit(1)

        stats = calculate_statistics(distances)

        print(f"\nResults for '{method}' method:")
        print("=" * 60)
        print(f"Total rows processed: {total_count}")
        print(f"Valid data points: {valid_count}")
        print(f"Skipped rows: {total_count - valid_count}")
        print("-" * 60)
        print(f"Average error distance: {stats['average']:.4f} pixels")
        print(f"Minimum error distance: {stats['min']:.4f} pixels")
        print(f"Maximum error distance: {stats['max']:.4f} pixels")
        print(f"Standard deviation: {stats['std_dev']:.4f} pixels")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()