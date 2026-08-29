"""Create a spatially balanced target-disjoint JuDo1000 split.

The older `judo_1000_split_no_leakage` split shuffled 48 target points and
then took contiguous train/val/test chunks. That is target-disjoint, but with
only 7 validation targets it can put validation in a very different screen
region than test. This script keeps the target-disjoint property and searches
for validation/test target sets whose spatial and row-count statistics are
close to the full dataset.

It writes raw split CSVs with only target/origin/spread columns. Run
`generate_judo_baselines_no_leakage.py` afterwards to fit classical baselines
on the new train split only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


BASE_COLUMNS = ["target_x", "target_y", "origin_gaze_x", "origin_gaze_y", "spread"]


def load_source(source_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = source_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "origin_gaze_x" not in df.columns and "original_gaze_x" in df.columns:
            df = df.rename(columns={"original_gaze_x": "origin_gaze_x", "original_gaze_y": "origin_gaze_y"})
        missing = [c for c in BASE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        base = df[BASE_COLUMNS].copy()
        base["source_split"] = split
        frames.append(base)
    out = pd.concat(frames, ignore_index=True)
    out["target_x"] = out["target_x"].astype(float)
    out["target_y"] = out["target_y"].astype(float)
    return out


def target_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["target_x", "target_y"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["target_y", "target_x"])
        .reset_index(drop=True)
    )
    table["target_id"] = np.arange(len(table))
    return table


def weighted_stats(table: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    sub = table.iloc[idx]
    xy = sub[["target_x", "target_y"]].to_numpy(float)
    w = sub["n"].to_numpy(float)
    w = w / w.sum()
    mean = (xy * w[:, None]).sum(axis=0)
    var = ((xy - mean) ** 2 * w[:, None]).sum(axis=0)
    return np.concatenate([mean, np.sqrt(var)])


def split_score(
    table: pd.DataFrame,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    full_stats: np.ndarray,
    full_rows: int,
    n_val_targets: int,
    n_test_targets: int,
) -> float:
    all_idx = np.arange(len(table))
    holdout = np.concatenate([val_idx, test_idx])
    train_idx = np.setdiff1d(all_idx, holdout, assume_unique=False)

    val_stats = weighted_stats(table, val_idx)
    test_stats = weighted_stats(table, test_idx)
    train_stats = weighted_stats(table, train_idx)

    scale = np.array([640.0, 512.0, 320.0, 256.0])
    val_diff = ((val_stats - full_stats) / scale) ** 2
    test_diff = ((test_stats - full_stats) / scale) ** 2
    train_diff = ((train_stats - full_stats) / scale) ** 2
    vt_diff = ((val_stats - test_stats) / scale) ** 2

    row_ratios = np.array(
        [
            table.iloc[train_idx]["n"].sum() / full_rows,
            table.iloc[val_idx]["n"].sum() / full_rows,
            table.iloc[test_idx]["n"].sum() / full_rows,
        ]
    )
    target_ratios = np.array(
        [
            (len(table) - n_val_targets - n_test_targets) / len(table),
            n_val_targets / len(table),
            n_test_targets / len(table),
        ]
    )
    ratio_penalty = ((row_ratios - target_ratios) ** 2).sum()

    # Encourage each held-out split to cover more than a tiny local patch.
    range_penalty = 0.0
    for idx in [val_idx, test_idx]:
        sub = table.iloc[idx]
        x_span = sub["target_x"].max() - sub["target_x"].min()
        y_span = sub["target_y"].max() - sub["target_y"].min()
        range_penalty += max(0.0, 300.0 - x_span) / 300.0
        range_penalty += max(0.0, 300.0 - y_span) / 300.0

    return float(
        6.0 * (val_diff[:2].sum() + test_diff[:2].sum())
        + 3.0 * (val_diff[2:].sum() + test_diff[2:].sum())
        + 1.5 * train_diff.sum()
        + 1.0 * vt_diff.sum()
        + 30.0 * ratio_penalty
        + 0.5 * range_penalty
    )


def search_split(
    table: pd.DataFrame,
    *,
    n_val_targets: int,
    n_test_targets: int,
    seed: int,
    candidates: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(table)
    all_idx = np.arange(n)
    full_stats = weighted_stats(table, all_idx)
    full_rows = int(table["n"].sum())

    best_val = None
    best_test = None
    best_score = float("inf")

    for _ in range(candidates):
        perm = rng.permutation(n)
        val_idx = np.sort(perm[:n_val_targets])
        test_idx = np.sort(perm[n_val_targets : n_val_targets + n_test_targets])
        score = split_score(table, val_idx, test_idx, full_stats, full_rows, n_val_targets, n_test_targets)
        if score < best_score:
            best_score = score
            best_val = val_idx
            best_test = test_idx

    assert best_val is not None and best_test is not None
    return best_val, best_test, best_score


def split_dataframe(df: pd.DataFrame, table: pd.DataFrame, val_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, pd.DataFrame]:
    val_targets = set(map(tuple, table.iloc[val_idx][["target_x", "target_y"]].to_numpy(float)))
    test_targets = set(map(tuple, table.iloc[test_idx][["target_x", "target_y"]].to_numpy(float)))

    def target_tuple(frame: pd.DataFrame) -> Iterable[Tuple[float, float]]:
        return zip(frame["target_x"].astype(float), frame["target_y"].astype(float))

    mask_val = pd.Series([t in val_targets for t in target_tuple(df)], index=df.index)
    mask_test = pd.Series([t in test_targets for t in target_tuple(df)], index=df.index)
    mask_train = ~(mask_val | mask_test)

    out = {
        "train": df.loc[mask_train, BASE_COLUMNS].copy(),
        "val": df.loc[mask_val, BASE_COLUMNS].copy(),
        "test": df.loc[mask_test, BASE_COLUMNS].copy(),
    }
    for split_df in out.values():
        split_df.sort_values(["target_y", "target_x"]).reset_index(drop=True, inplace=True)
    return out


def split_diagnostics(table: pd.DataFrame, split_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    full_stats = weighted_stats(table, np.arange(len(table)))
    rows = []
    for split, idx in split_map.items():
        sub = table.iloc[idx]
        stats = weighted_stats(table, idx)
        rows.append(
            {
                "split": split,
                "num_targets": int(len(idx)),
                "num_rows": int(sub["n"].sum()),
                "target_x_mean": stats[0],
                "target_y_mean": stats[1],
                "target_x_std": stats[2],
                "target_y_std": stats[3],
                "target_x_mean_delta_full": stats[0] - full_stats[0],
                "target_y_mean_delta_full": stats[1] - full_stats[1],
                "target_x_min": float(sub["target_x"].min()),
                "target_x_max": float(sub["target_x"].max()),
                "target_y_min": float(sub["target_y"].min()),
                "target_y_max": float(sub["target_y"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a spatially balanced target-disjoint JuDo1000 split")
    parser.add_argument("--source-dir", type=Path, default=Path("data/prepared/judo_1000_split_no_leakage"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared/judo_1000_target_balanced_no_leakage"))
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--candidates", type=int, default=200_000)
    parser.add_argument("--val-targets", type=int, default=7)
    parser.add_argument("--test-targets", type=int, default=8)
    args = parser.parse_args()

    df = load_source(args.source_dir)
    table = target_table(df)
    if args.val_targets + args.test_targets >= len(table):
        raise ValueError("val-targets + test-targets must be smaller than total target count")

    val_idx, test_idx, score = search_split(
        table,
        n_val_targets=args.val_targets,
        n_test_targets=args.test_targets,
        seed=args.seed,
        candidates=args.candidates,
    )
    all_idx = np.arange(len(table))
    train_idx = np.setdiff1d(all_idx, np.concatenate([val_idx, test_idx]), assume_unique=False)

    splits = split_dataframe(df, table, val_idx, test_idx)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_df in splits.items():
        split_df.to_csv(args.output_dir / f"{split}.csv", index=False)

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    diag = split_diagnostics(table, split_map)
    diag.to_csv(args.output_dir / "split_diagnostics.csv", index=False)

    target_rows = []
    for split, idx in split_map.items():
        sub = table.iloc[idx].copy()
        sub.insert(0, "split", split)
        target_rows.append(sub)
    pd.concat(target_rows, ignore_index=True).to_csv(args.output_dir / "split_targets.csv", index=False)

    metadata = {
        "split_type": "target_disjoint_spatial_balanced",
        "source_dir": str(args.source_dir),
        "seed": args.seed,
        "candidates": args.candidates,
        "score": score,
        "total_samples": int(len(df)),
        "unique_targets": int(len(table)),
        "splits": {
            split: {
                "num_target_points": int(splits[split][["target_x", "target_y"]].drop_duplicates().shape[0]),
                "num_samples": int(len(splits[split])),
            }
            for split in ["train", "val", "test"]
        },
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote balanced split to {args.output_dir}")
    print(json.dumps(metadata, indent=2))
    print("\nDiagnostics:")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
