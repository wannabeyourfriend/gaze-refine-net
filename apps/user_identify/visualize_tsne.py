#!/usr/bin/env python3
"""
Visualize session embeddings using t-SNE.

Usage:
    python visualize_tsne.py outputs/checkpoints/best_model.pt
    python visualize_tsne.py outputs/checkpoints/best_model.pt --perplexity 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.dataset import create_datasets
from src.model import build_model


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract session embeddings from model."""
    model.eval()

    all_embeddings = []
    all_labels = []

    for features, mask, labels in dataloader:
        features = features.to(device)
        mask = mask.to(device)

        # Get embeddings (normalized)
        _, embeddings = model(features, mask, return_embedding=True)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def plot_tsne(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    user_names: list,
    split_name: str,
    output_path: Path,
) -> None:
    """Create t-SNE visualization."""
    plt.figure(figsize=(10, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(user_names)))

    # Plot each user
    for user_id, user_name in enumerate(user_names):
        mask = labels == user_id
        if mask.sum() == 0:
            continue

        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[user_id]],
            label=f"{user_name} (n={mask.sum()})",
            s=100,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title(f"Session Embeddings - {split_name}\n(t-SNE Visualization)", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings with t-SNE")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "default.yaml",
        help="Path to config file",
    )
    parser.add_argument("--perplexity", type=float, default=5, help="t-SNE perplexity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve data path
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / config["data"]["csv_path"]

    # Create datasets
    feature_flags = config.get("features", {})
    train_dataset, val_dataset, test_dataset, user_to_id, num_features = create_datasets(
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

    # User names
    id_to_user = {v: k for k, v in user_to_id.items()}
    user_names = [id_to_user[i] for i in range(len(user_to_id))]

    print(f"Users: {user_names}")

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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Extract embeddings from all splits
    print("\nExtracting embeddings...")
    train_emb, train_labels = extract_embeddings(model, train_loader, device)
    val_emb, val_labels = extract_embeddings(model, val_loader, device)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)

    # Combine all embeddings
    all_emb = np.concatenate([train_emb, val_emb, test_emb])
    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    all_splits = (
        ["train"] * len(train_emb)
        + ["val"] * len(val_emb)
        + ["test"] * len(test_emb)
    )

    print(f"Total sessions: {len(all_emb)}")
    print(f"Embedding dimension: {all_emb.shape[1]}")

    # Fit t-SNE on all data
    print(f"\nFitting t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, len(all_emb) - 1),
        random_state=args.seed,
        max_iter=500,
    )
    all_emb_2d = tsne.fit_transform(all_emb)

    # Split back
    n_train = len(train_emb)
    n_val = len(val_emb)
    train_emb_2d = all_emb_2d[:n_train]
    val_emb_2d = all_emb_2d[n_train : n_train + n_val]
    test_emb_2d = all_emb_2d[n_train + n_val :]

    # Plot all data combined
    plot_tsne(
        all_emb_2d,
        all_labels,
        user_names,
        "All Sessions",
        args.output_dir / "tsne_all.png",
    )

    # Plot with split markers
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(user_names)))
    markers = {"train": "o", "val": "s", "test": "^"}

    for user_id, user_name in enumerate(user_names):
        for split_name, (emb_2d, labels) in [
            ("train", (train_emb_2d, train_labels)),
            ("val", (val_emb_2d, val_labels)),
            ("test", (test_emb_2d, test_labels)),
        ]:
            mask = labels == user_id
            if mask.sum() == 0:
                continue

            label = f"{user_name}" if split_name == "train" else None
            plt.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[colors[user_id]],
                label=label,
                marker=markers[split_name],
                s=120 if split_name == "test" else 80,
                alpha=0.8 if split_name == "test" else 0.5,
                edgecolors="black" if split_name == "test" else "white",
                linewidth=1 if split_name == "test" else 0.5,
            )

    # Add legend for markers
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", label="Train", markersize=8, linestyle="None"),
        Line2D([0], [0], marker="s", color="gray", label="Val", markersize=8, linestyle="None"),
        Line2D([0], [0], marker="^", color="gray", label="Test", markersize=10, linestyle="None"),
    ]

    # First legend for users
    leg1 = plt.legend(loc="upper left", fontsize=9, title="Users")
    plt.gca().add_artist(leg1)

    # Second legend for splits
    plt.legend(handles=legend_elements, loc="upper right", fontsize=9, title="Split")

    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title("Session Embeddings by User\n(t-SNE, markers indicate train/val/test)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = args.output_dir / "tsne_by_split.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
