"""Leave-One-Subject-Out (LOSO) evaluation for the all_trials_split data.

Uses the existing per-trial-fitted baselines (already in the CSVs) and
runs the v2 honest pipeline subject-by-subject. For each held-out
subject, the network is trained on the remaining 11 subjects' rows and
evaluated on the held-out subject's rows.

This is the strongest honest evaluation we can extract without
recollecting data: it satisfies the reviewer's concern about
subject-disjoint splits (the network never sees test-subject samples
during training) while leveraging the per-trial baseline calibrations
that are already correctly fit (each trial fits its own baselines on
its own 18 calibration points).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from refine_v2 import V2Config, train_v2


def loso_evaluate(
    csv_path: str,
    cfg_template: V2Config,
    *,
    subjects: List[str] | None = None,
    save_dir: Path | None = None,
) -> List[Dict]:
    """Leave-one-subject-out evaluation.

    For each subject s:
      train: rows where subject != s   (concat of trials)
      val:   10% of train rows held out (random by trial)
      test:  rows where subject == s
    """
    df_all = pd.read_csv(csv_path)
    all_subs = sorted(df_all["subject"].unique())
    subjects = subjects or all_subs
    print(f"Subjects ({len(subjects)}): {subjects}")

    results = []
    for held in subjects:
        sub_df = df_all[df_all["subject"] == held]
        rest_df = df_all[df_all["subject"] != held]
        if len(sub_df) < 5 or len(rest_df) < 50:
            print(f"  skip {held}: too few rows ({len(sub_df)} held, {len(rest_df)} rest)")
            continue

        # Random val split from rest, by trial
        rng = np.random.default_rng(cfg_template.seed)
        rest_df = rest_df.copy()
        rest_df["trial_id"] = rest_df["subject"] + "|" + rest_df["timestamp"]
        trials = sorted(rest_df["trial_id"].unique())
        rng.shuffle(trials)
        n_val = max(1, int(len(trials) * 0.10))
        val_trials = set(trials[:n_val])
        val_df = rest_df[rest_df["trial_id"].isin(val_trials)].drop(columns=["trial_id"])
        train_df = rest_df[~rest_df["trial_id"].isin(val_trials)].drop(columns=["trial_id"])

        # Materialize temp CSVs
        if save_dir is None:
            save_dir = Path("/tmp/loso_csvs")
        save_dir.mkdir(parents=True, exist_ok=True)
        train_p = save_dir / f"_loso_train_{held}.csv"
        val_p = save_dir / f"_loso_val_{held}.csv"
        test_p = save_dir / f"_loso_test_{held}.csv"
        train_df.to_csv(train_p, index=False)
        val_df.to_csv(val_p, index=False)
        sub_df.to_csv(test_p, index=False)

        cfg = V2Config(
            train_csv=str(train_p),
            val_csv=str(val_p),
            test_csv=str(test_p),
            **{k: v for k, v in cfg_template.__dict__.items() if k not in {"train_csv", "val_csv", "test_csv"}},
        )

        # Reference: classical baseline performance on this subject
        sim_err = np.linalg.norm(
            sub_df[["pred_sim_rbf_multiquadric_s2.0_x", "pred_sim_rbf_multiquadric_s2.0_y"]].values
            - sub_df[["target_x", "target_y"]].values,
            axis=1,
        )
        baseline_l2 = float(sim_err.mean())

        out = train_v2(cfg, verbose=False)
        result = {
            "held_subject": held,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(sub_df),
            "baseline_l2": baseline_l2,
            "best_val_l2": out["best_val_l2"],
            "best_epoch": out["best_epoch"],
            **{f"test_{k}": v for k, v in out["test"].items()},
        }
        result["delta_vs_baseline"] = result["test_l2_mean"] - baseline_l2
        results.append(result)
        print(f"  {held:25s}  base={baseline_l2:6.2f}  ours={result['test_l2_mean']:6.2f}  "
              f"delta={result['delta_vs_baseline']:+6.2f}  med={result['test_l2_median']:.1f}")
    return results
