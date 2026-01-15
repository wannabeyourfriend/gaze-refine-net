#!/usr/bin/env python3
"""
Evaluation script for user identification model.

Usage:
    python eval.py outputs/checkpoints/best_model.pt --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import create_datasets
from src.model import build_model


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    user_names: list,
) -> dict:
    """Detailed evaluation of the model."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for features, mask, labels in dataloader:
        features = features.to(device)
        mask = mask.to(device)

        logits, _ = model(features, mask, return_embedding=False)
        probs = torch.softmax(logits, dim=-1)

        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()

    # Get unique labels that appear in predictions/labels
    unique_labels = sorted(set(all_labels) | set(all_preds))
    filtered_user_names = [user_names[i] for i in unique_labels]

    # Per-class metrics
    report = classification_report(
        all_labels,
        all_preds,
        labels=unique_labels,
        target_names=filtered_user_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    # Top-k accuracy
    top3_correct = 0
    for i, (label, probs) in enumerate(zip(all_labels, all_probs)):
        top3_preds = np.argsort(probs)[-3:]
        if label in top3_preds:
            top3_correct += 1
    top3_acc = top3_correct / len(all_labels)

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "filtered_user_names": filtered_user_names,
    }


def print_evaluation_results(results: dict, user_names: list) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"Top-3 Accuracy: {results['top3_accuracy']:.4f} ({results['top3_accuracy']*100:.1f}%)")

    print("\n" + "-" * 60)
    print("Per-User Performance:")
    print("-" * 60)
    print(f"{'User':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    # Use filtered user names from results
    display_names = results.get("filtered_user_names", user_names)
    report = results["classification_report"]
    for user in display_names:
        if user in report:
            m = report[user]
            print(
                f"{user:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                f"{m['f1-score']:<12.3f} {int(m['support']):<10}"
            )

    print("-" * 60)
    m = report["weighted avg"]
    print(
        f"{'Weighted Avg':<20} {m['precision']:<12.3f} {m['recall']:<12.3f} "
        f"{m['f1-score']:<12.3f} {int(m['support']):<10}"
    )

    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)

    cm = results["confusion_matrix"]
    max_name_len = max(len(name) for name in display_names)

    # Header
    header = " " * (max_name_len + 2) + " ".join(f"{name[:6]:>6}" for name in display_names)
    print(header)

    # Rows
    for i, name in enumerate(display_names):
        row = f"{name:<{max_name_len}}  " + " ".join(f"{cm[i, j]:>6}" for j in range(len(display_names)))
        print(row)

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate user identification model")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "default.yaml",
        help="Path to config file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve data path
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / config["data"]["csv_path"]

    # Create datasets (with same split as training)
    feature_flags = config.get("features", {})
    _, _, test_dataset, user_to_id, num_features = create_datasets(
        csv_path=csv_path,
        min_sessions_per_user=config["data"].get("min_sessions_per_user", 2),
        min_points_per_session=config["data"].get("min_points_per_session", 10),
        test_ratio=config["data"].get("test_ratio", 0.2),
        val_ratio=config["data"].get("val_ratio", 0.1),
        max_points=24,
        seed=args.seed,
        use_original_gaze=feature_flags.get("use_original_gaze", True),
        use_target=feature_flags.get("use_target", True),
        use_sim_rbf=feature_flags.get("use_sim_rbf", True),
        use_spread=feature_flags.get("use_spread", True),
        use_residuals=feature_flags.get("use_residuals", True),
    )

    # User names in order
    id_to_user = {v: k for k, v in user_to_id.items()}
    user_names = [id_to_user[i] for i in range(len(user_to_id))]

    print(f"Test sessions: {len(test_dataset)}")
    print(f"Users: {user_names}")

    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Build model
    model_config = config.get("model", {})
    model = build_model(
        input_dim=num_features,
        num_classes=len(user_to_id),
        config=model_config,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded model from: {args.checkpoint}")

    # Evaluate
    results = evaluate_model(model, test_loader, device, user_names)
    print_evaluation_results(results, user_names)


if __name__ == "__main__":
    main()
