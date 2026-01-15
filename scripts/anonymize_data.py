#!/usr/bin/env python3
"""
Anonymize subject names in CSV data files.
Replaces real names with anonymous IDs (e.g., "subject_001").
"""

import os
import re
from pathlib import Path
import pandas as pd
from typing import Dict


def get_subject_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a mapping from real subject names to anonymous IDs.

    Args:
        df: DataFrame containing subject_name column

    Returns:
        Dictionary mapping original names to anonymous IDs
    """
    if 'subject_name' not in df.columns:
        return {}

    unique_subjects = sorted(df['subject_name'].unique())
    mapping = {
        name: f'subject_{i:03d}'
        for i, name in enumerate(unique_subjects, 1)
    }
    return mapping


def anonymize_csv_file(file_path: Path, dry_run: bool = True) -> Dict[str, str]:
    """
    Anonymize subject names in a CSV file.

    Args:
        file_path: Path to the CSV file
        dry_run: If True, don't actually modify files

    Returns:
        Dictionary of name mappings if any changes were made
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return {}

    if 'subject_name' not in df.columns:
        return {}

    mapping = get_subject_mapping(df)

    if not mapping:
        return {}

    print(f"\n{file_path.relative_to(file_path.parents[2])}:")
    print(f"  Found subjects: {list(mapping.keys())}")
    print(f"  Mapped to: {list(mapping.values())}")

    if not dry_run:
        df['subject_name'] = df['subject_name'].map(mapping)
        df.to_csv(file_path, index=False)
        print(f"  ✓ Updated")
    else:
        print(f"  [Dry run - would update]")

    return mapping


def find_csv_files(data_dir: Path) -> list[Path]:
    """
    Find all CSV files in the data directory.

    Args:
        data_dir: Root data directory

    Returns:
        List of CSV file paths
    """
    csv_files = []
    for csv_file in data_dir.rglob('*.csv'):
        # Skip checkpoint files and temporary files
        if any(x in str(csv_file) for x in ['checkpoints', '.git', '__pycache__']):
            continue
        csv_files.append(csv_file)
    return sorted(csv_files)


def main():
    """Main function to anonymize all CSV files."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Anonymize subject names in CSV files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm before making changes'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return 1

    csv_files = find_csv_files(data_dir)

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return 0

    print(f"Found {len(csv_files)} CSV files to check")
    print("=" * 60)

    # Track all unique subjects across all files
    all_subjects = set()
    files_with_changes = []

    for csv_file in csv_files:
        mapping = anonymize_csv_file(csv_file, dry_run=True)
        if mapping:
            all_subjects.update(mapping.keys())
            files_with_changes.append(csv_file)

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Total CSV files: {len(csv_files)}")
    print(f"  Files with subject names: {len(files_with_changes)}")
    print(f"  Unique subjects found: {len(all_subjects)}")
    print(f"  Subject names: {sorted(all_subjects)}")

    if args.dry_run:
        print("\n[Dry run mode - no files were modified]")
        print("Run without --dry-run to apply changes")
        return 0

    if not args.confirm:
        response = input("\nProceed with anonymization? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled")
            return 0

    print("\nApplying changes...")
    for csv_file in files_with_changes:
        anonymize_csv_file(csv_file, dry_run=False)

    print("\n✓ Anonymization complete")
    return 0


if __name__ == '__main__':
    exit(main())