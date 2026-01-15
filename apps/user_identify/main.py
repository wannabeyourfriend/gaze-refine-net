#!/usr/bin/env python3
"""
Main training script for user identification from gaze patterns.

Usage:
    python main.py --config config/default.yaml
    python main.py  # Uses default config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import create_datasets
from src.model import build_model
from src.trainer import Trainer, TrainingConfig


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train user identification model")
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
    print(f"Loaded config from {args.config}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve data path
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / config["data"]["csv_path"]
    print(f"Loading data from: {csv_path}")

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

    num_classes = len(user_to_id)
    print(f"\nDataset summary:")
    print(f"  Features per point: {num_features}")
    print(f"  Number of users: {num_classes}")
    print(f"  Users: {list(user_to_id.keys())}")
    print(f"  Train sessions: {len(train_dataset)}")
    print(f"  Val sessions: {len(val_dataset)}")
    print(f"  Test sessions: {len(test_dataset)}")

    # Create data loaders
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Build model
    model_config = config.get("model", {})
    model = build_model(
        input_dim=num_features,
        num_classes=num_classes,
        config=model_config,
    )
    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Training config
    scheduler_cfg = train_cfg.get("scheduler", {})
    early_stop_cfg = train_cfg.get("early_stopping", {})

    training_config = TrainingConfig(
        batch_size=batch_size,
        epochs=train_cfg.get("epochs", 100),
        learning_rate=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
        use_contrastive_loss=train_cfg.get("use_contrastive_loss", True),
        contrastive_weight=train_cfg.get("contrastive_weight", 0.5),
        temperature=train_cfg.get("temperature", 0.1),
        warmup_epochs=scheduler_cfg.get("warmup_epochs", 5),
        scheduler_type=scheduler_cfg.get("type", "cosine"),
        patience=early_stop_cfg.get("patience", 15),
        min_delta=early_stop_cfg.get("min_delta", 0.001),
    )

    # Create checkpoint directory
    output_cfg = config.get("output", {})
    checkpoint_dir = Path(__file__).parent / output_cfg.get(
        "checkpoint_dir", "outputs/checkpoints"
    ).replace("apps/user_identify/", "")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    print("\nStarting training...")
    print("=" * 80)
    history = trainer.train(train_loader, val_loader, verbose=True)
    print("=" * 80)

    # Load best model and evaluate on test set
    trainer.load_checkpoint("best_model.pt")
    test_loss, test_acc = trainer.evaluate(test_loader)

    print(f"\nFinal Results:")
    print(f"  Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")

    # Save results
    results = {
        "num_users": num_classes,
        "users": list(user_to_id.keys()),
        "num_features": num_features,
        "train_sessions": len(train_dataset),
        "val_sessions": len(val_dataset),
        "test_sessions": len(test_dataset),
        "best_val_acc": trainer.best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
    }

    results_path = checkpoint_dir / "results.yaml"
    with open(results_path, "w") as f:
        # Convert numpy/tensor values to Python types
        def convert(obj):
            if isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif hasattr(obj, "item"):
                return obj.item()
            return obj

        yaml.dump(convert(results), f, default_flow_style=False)

    print(f"\nResults saved to: {results_path}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
