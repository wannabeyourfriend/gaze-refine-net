#!/usr/bin/env python3
"""
Analyze the filtered_trials_summary markdown file.
This script parses the embedded CSV data without using pandas.
"""

import re
import ast
from collections import defaultdict

def analyze_markdown_file(md_path):
    """Analyze the markdown file with embedded CSV data."""

    with open(md_path, 'r') as f:
        content = f.read()

    print("=" * 60)
    print("FILTERED TRIALS SUMMARY ANALYSIS")
    print("=" * 60)
    print(f"\nFile: {md_path}")
    print(f"Size: {len(content) / 1024:.1f} KB\n")

    # Extract CSV content from markdown code block
    csv_match = re.search(r'```csv\n(.*?)\n```', content, re.DOTALL)
    if not csv_match:
        print("❌ No CSV code block found in markdown")
        return

    csv_content = csv_match.group(1)
    lines = csv_content.split('\n')

    print("📋 FILE STRUCTURE")
    print("-" * 40)
    print(f"Format: Markdown with embedded CSV")
    print(f"CSV lines: {len(lines)}")
    print(f"Header: {lines[0][:80]}...")

    # Parse header
    headers = [h.strip() for h in lines[0].split(',')]
    print(f"\n📊 Columns ({len(headers)}):")
    for i, h in enumerate(headers, 1):
        print(f"  {i:2d}. {h}")

    # The actual CSV data is in a corresponding CSV file
    # The .md file just shows a preview
    csv_path = md_path.replace('.md', '.csv')
    print(f"\n📁 Full data location: {csv_path}")

    # Try to get basic info about the CSV file without opening it fully
    import os
    if os.path.exists(csv_path):
        csv_size = os.path.getsize(csv_path)
        print(f"CSV file size: {csv_size / 1024 / 1024:.1f} MB")

        # Count lines efficiently
        print("\n🔍 Counting rows in full CSV file...")
        line_count = 0
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Count first 100 lines
                    line_count = i + 1
                    break
            else:
                line_count = i + 1

        # Fast line count using subprocess
        import subprocess
        result = subprocess.run(['wc', '-l', csv_path], capture_output=True, text=True)
        total_lines = int(result.stdout.split()[0]) if result.returncode == 0 else 0

        print(f"Total rows (including header): {total_lines:,}")
        print(f"Data rows: {total_lines - 1:,}")

    # Parse the preview data from markdown
    print("\n" + "=" * 60)
    print("PREVIEW DATA FROM MARKDOWN")
    print("=" * 60)

    if len(lines) > 1:
        # Extract first few rows from the preview
        # The data might be all on line 1 with embedded JSON
        data_line = lines[1]

        # Use regex to extract structured data
        # Pattern: file_prefix,trial_id,point_num,target_x,target_y,mean_x,mean_y,distance,num_raw_points
        pattern = r'(\d+_\d+),(\d+),(\d+),([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+),(\d+)'
        matches = list(re.finditer(pattern, data_line))

        if matches:
            print(f"\nPreview shows {len(matches)} rows\n")

            print("📍 SAMPLE ROWS:")
            for i, match in enumerate(matches[:5], 1):
                print(f"\n  Row {i}:")
                print(f"    Trial: {match.group(1)}")
                print(f"    Point: {match.group(3)}")
                print(f"    Target: ({match.group(4)}, {match.group(5)})")
                print(f"    Mean Gaze: ({match.group(6)}, {match.group(7)})")
                print(f"    Error: {match.group(8)} px")
                print(f"    Raw samples: {match.group(9)}")

            # Statistics from preview
            file_prefixes = set(m.group(1) for m in matches)
            trial_ids = set(m.group(2) for m in matches)
            point_nums = [int(m.group(3)) for m in matches]
            distances = [float(m.group(8)) for m in matches]

            print(f"\n📈 PREVIEW STATISTICS:")
            print(f"  Unique trials: {len(file_prefixes)}")
            print(f"  Trial IDs: {sorted(trial_ids)}")
            print(f"  Point numbers: {sorted(set(point_nums))}")
            print(f"  Distance range: {min(distances):.2f} - {max(distances):.2f} px")

    print("\n" + "=" * 60)
    print("NOTES")
    print("=" * 60)
    print("• The .md file contains only a preview of the full dataset")
    print("• Full data (835 MB) is in the corresponding .csv file")
    print("• Each row contains aggregated calibration point data")
    print("• raw_data field contains detailed gaze tracking history")
    print("• Use the CSV file with pandas for full analysis")
    print("=" * 60)


if __name__ == "__main__":
    md_path = "/Users/admin/Codebase/NRMBC-Neural-Refined-Model-Based-Gazing-Point-Calibration/data/raw/all/filtered_trials_summary_first_tenth.md"
    analyze_markdown_file(md_path)