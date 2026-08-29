#!/usr/bin/env python3
"""
Anonymize subject names in CSV data files.
Replaces real names with anonymous IDs (e.g., "subject_001").

The mapping is built once across every file so that a given person keeps
the same ID everywhere; a per-file mapping would silently break the
subject-disjoint splits that join train/val/test by subject.
"""

import json
import re
from pathlib import Path
import pandas as pd
from typing import Dict, List

# Columns that have held a real identifier at some point in this repo's
# history. 'subject' is the current spelling; 'subject_name' appears in the
# two legacy standard.csv files.
SUBJECT_COLUMNS = ('subject', 'subject_name')

MAPPING_FILENAME = 'subject_mapping.json'

# Identifiers that are already pseudonymous and must be left alone.
# 'judo_N' are JuDo1000's own subject IDs: they are an external contract —
# rewriting them would break every join against the public dataset and against
# the per-fold LOSO tables. 'subject_NNN' are IDs this script already assigned.
PSEUDONYM_RE = re.compile(r'^(subject_\d+|judo_\d+)$')


def is_pseudonymous(value: str) -> bool:
    """True if the identifier carries no real-world name."""
    return bool(PSEUDONYM_RE.match(value))


def subject_column(df: pd.DataFrame) -> str | None:
    """Return the identifier column present in df, if any."""
    for col in SUBJECT_COLUMNS:
        if col in df.columns:
            return col
    return None


def read_subjects(file_path: Path) -> tuple[str | None, set]:
    """
    Read the identifier column and its distinct values from one CSV.

    Only that one column is parsed — these files run to 148 columns and
    hundreds of MB, and the scan pass touches every one of them.
    """
    try:
        header = pd.read_csv(file_path, nrows=0)
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return None, set()

    col = subject_column(header)
    if col is None:
        return None, set()

    try:
        values = pd.read_csv(file_path, usecols=[col])[col]
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return None, set()

    return col, set(values.dropna().astype(str).unique())


def build_global_mapping(csv_files: List[Path]) -> Dict[str, str]:
    """
    Build one name -> anonymous ID mapping covering every file.

    Values that are already anonymized (subject_NNN) map to themselves, so
    re-running the script is idempotent and does not renumber anyone.
    """
    all_names = set()
    for csv_file in csv_files:
        _, names = read_subjects(csv_file)
        all_names.update(names)

    already_anon = {n for n in all_names if is_pseudonymous(n)}
    to_map = sorted(all_names - already_anon)

    mapping = {n: n for n in sorted(already_anon)}
    taken = {int(n.split('_')[1]) for n in already_anon
             if n.startswith('subject_') and n.split('_')[1].isdigit()}

    next_id = 1
    for name in to_map:
        while next_id in taken:
            next_id += 1
        mapping[name] = f'subject_{next_id:03d}'
        taken.add(next_id)

    return mapping


def anonymize_csv_file(file_path: Path, mapping: Dict[str, str],
                       dry_run: bool = True) -> set:
    """
    Apply the global mapping to one CSV file.

    Returns the set of real names that were found in this file.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return set()

    col = subject_column(df)
    if col is None:
        return set()

    values = df[col].dropna().astype(str)
    found = {v for v in values.unique() if not is_pseudonymous(v)}
    if not found:
        return set()

    if not dry_run:
        df[col] = df[col].astype(str).map(lambda v: mapping.get(v, v))
        df.to_csv(file_path, index=False)

    return found


def find_csv_files(data_dirs: List[Path]) -> List[Path]:
    """
    Find all CSV files under the given roots.

    Args:
        data_dirs: Root directories to scan

    Returns:
        List of CSV file paths
    """
    csv_files = []
    for data_dir in data_dirs:
        for csv_file in data_dir.rglob('*.csv'):
            # Skip checkpoint files and temporary files
            if any(x in str(csv_file) for x in ['checkpoints', '.git', '__pycache__']):
                continue
            csv_files.append(csv_file)
    return sorted(set(csv_files))


def main():
    """Main function to anonymize all CSV files."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Anonymize subject names in CSV files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        nargs='+',
        default=['data'],
        help='Path(s) to data directory (default: data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip the interactive prompt before making changes'
    )
    parser.add_argument(
        '--mapping-out',
        type=str,
        default=MAPPING_FILENAME,
        help=f'Where to write the name mapping (default: {MAPPING_FILENAME}). '
             'Keep this file out of any public release.'
    )

    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dir]
    missing = [d for d in data_dirs if not d.exists()]
    if missing:
        print(f"Error: Directory {missing[0]} does not exist")
        return 1

    csv_files = find_csv_files(data_dirs)

    if not csv_files:
        print(f"No CSV files found in {', '.join(str(d) for d in data_dirs)}")
        return 0

    print(f"Found {len(csv_files)} CSV files to check")
    print("=" * 60)

    mapping = build_global_mapping(csv_files)
    real_names = {k: v for k, v in mapping.items() if k != v}

    files_with_changes = []
    for csv_file in csv_files:
        _, names = read_subjects(csv_file)
        if any(not is_pseudonymous(n) for n in names):
            files_with_changes.append(csv_file)

    print(f"Summary:")
    print(f"  Total CSV files: {len(csv_files)}")
    print(f"  Files still holding real names: {len(files_with_changes)}")
    print(f"  Unique subjects found: {len(real_names)}")
    for name, anon in sorted(real_names.items(), key=lambda kv: kv[1]):
        print(f"    {name!r} -> {anon}")

    if args.dry_run:
        print("\n[Dry run mode - no files were modified]")
        print("Run without --dry-run to apply changes")
        return 0

    if not files_with_changes:
        print("\n✓ Nothing to do — every file is already anonymized")
        return 0

    if not args.confirm:
        response = input("\nProceed with anonymization? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled")
            return 0

    print("\nApplying changes...")
    for csv_file in files_with_changes:
        anonymize_csv_file(csv_file, mapping, dry_run=False)
        print(f"  ✓ {csv_file}")

    mapping_path = Path(args.mapping_out)
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"\n✓ Anonymization complete")
    print(f"  Mapping written to {mapping_path} — do NOT commit or publish this file")
    return 0


if __name__ == '__main__':
    exit(main())
