from __future__ import annotations
"""
Train and evaluate the neural-refine gaze correction network.

Usage examples
--------------
Train with the default config:
    python apps/neural-refine/main.py

Override the config path:
    python apps/neural-refine/main.py --config path/to/config.yaml
"""
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
import pandas as pd

from src.model import GazeDataset, build_model


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (base / p).resolve()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    cfg: dict, model_type: str, coordinate_scale: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base = Path(__file__).resolve().parent
    data_cfg = cfg["data"]

    train_ds = GazeDataset(
        resolve_path(base, data_cfg["train_data_path"]),
        coordinate_scale=coordinate_scale,
        normalize=data_cfg.get("normalize", True),
        model_type=model_type,
    )
    val_ds = GazeDataset(
        resolve_path(base, data_cfg["val_data_path"]),
        coordinate_scale=coordinate_scale,
        normalize=data_cfg.get("normalize", True),
        model_type=model_type,
    )
    test_ds = GazeDataset(
        resolve_path(base, data_cfg["test_data_path"]),
        coordinate_scale=coordinate_scale,
        normalize=data_cfg.get("normalize", True),
        model_type=model_type,
    )

    return (
        DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 0),
        ),
        DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 0),
        ),
        DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 0),
        ),
    )


def build_optimizer(
    model: torch.nn.Module,
    cfg: dict,
) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    training_cfg = cfg["training"]
    optimizer_type = training_cfg.get("optimizer", "adam").lower()
    lr = training_cfg.get("learning_rate")
    weight_decay = training_cfg.get("weight_decay", 0.0)

    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        momentum = training_cfg.get("momentum", 0.9)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    
    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    training_cfg = cfg["training"]
    scheduler_type = training_cfg.get("scheduler", "none").lower()

    if scheduler_type == "none":
        return None

    if scheduler_type == "cosine":
        num_epochs = training_cfg["num_epochs"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=training_cfg.get("min_learning_rate", 1e-6),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_type}")


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    model: torch.nn.Module,
    loss_cfg: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse = F.mse_loss(predictions, targets)
    euclidean = torch.linalg.norm(predictions - targets, dim=1).mean()
    reg = torch.tensor(0.0, device=predictions.device)
    if loss_cfg.get("regularization_weight", 0.0) > 0:
        reg = sum((p ** 2).sum() for p in model.parameters())  # L2 over params
        reg = reg / sum(p.numel() for p in model.parameters())

    total = (
        loss_cfg.get("mse_weight", 1.0) * mse
        + loss_cfg.get("euclidean_weight", 1.0) * euclidean
        + loss_cfg.get("regularization_weight", 0.0) * reg
    )

    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_mse": float(mse.detach().cpu()),
        "loss_euclidean": float(euclidean.detach().cpu()),
        "loss_reg": float(reg.detach().cpu()),
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: dict,
    grad_clip: float | None = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss, _ = compute_loss(preds, targets, model, loss_cfg)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        batches += 1

    return {"train_loss": running_loss / max(1, batches)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict,
    coordinate_scale: float,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "count": 0}
    mae = torch.zeros(2, device=device)
    rmse = torch.zeros(2, device=device)
    l2_mean = 0.0

    # Respect dataset normalization; if inputs were already in pixels we
    # should not rescale again.
    scale = coordinate_scale if getattr(loader.dataset, "normalize", True) else 1.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        preds = model(inputs)
        loss, _ = compute_loss(preds, targets, model, loss_cfg)

        # Convert back to pixel units for interpretable metrics.
        pred_px = preds * scale
        target_px = targets * scale
        error_px = pred_px - target_px

        mae += error_px.abs().sum(dim=0)
        rmse += (error_px ** 2).sum(dim=0)
        l2_mean += torch.linalg.norm(error_px, dim=1).sum().item()

        running["loss"] += loss.item() * inputs.size(0)
        running["count"] += inputs.size(0)

    count = max(1, running["count"])
    mae = mae / count
    rmse = torch.sqrt(rmse / count)
    l2_mean = l2_mean / count

    return {
        "val_loss": running["loss"] / count,
        "mae_x_px": float(mae[0].cpu()),
        "mae_y_px": float(mae[1].cpu()),
        "rmse_x_px": float(rmse[0].cpu()),
        "rmse_y_px": float(rmse[1].cpu()),
        "l2_px": float(l2_mean),
    }


@torch.no_grad()
def export_predictions(
    split: str,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    coordinate_scale: float,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> Path:
    """
    Run inference on a dataset and save per-sample predictions to CSV.

    Columns follow the original data plus model outputs:
    target_x, target_y, original_gaze_x, original_gaze_y,
    pred_res_x, pred_res_y, pred_gaze_x, pred_gaze_y
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    rows = []

    scale = coordinate_scale if getattr(dataset, "normalize", True) else 1.0
    base_inputs_px = getattr(dataset, "sim_inputs_px", dataset.orig_inputs_px)

    offset = 0
    for inputs, _ in loader:
        bsz = inputs.size(0)
        inputs = inputs.to(device)
        preds = model(inputs)

        preds_px = preds * scale

        for i in range(bsz):
            idx = offset + i
            orig_px = dataset.orig_inputs_px[idx]
            sim_px = getattr(dataset, "sim_inputs_px", None)
            base_px = base_inputs_px[idx]
            target_px = dataset.targets_px[idx]
            pred_res_px = preds_px[i].cpu()
            pred_gaze_px = base_px + pred_res_px

            rows.append(
                {
                    "target_x": float(target_px[0]),
                    "target_y": float(target_px[1]),
                    "original_gaze_x": float(orig_px[0]),
                    "original_gaze_y": float(orig_px[1]),
                    "sim_rbf_gaze_x": float(sim_px[idx][0])
                    if sim_px is not None
                    else None,
                    "sim_rbf_gaze_y": float(sim_px[idx][1])
                    if sim_px is not None
                    else None,
                    "pred_res_x": float(pred_res_px[0]),
                    "pred_res_y": float(pred_res_px[1]),
                    "pred_gaze_x": float(pred_gaze_px[0]),
                    "pred_gaze_y": float(pred_gaze_px[1]),
                }
            )
        offset += bsz

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"predictions_{split}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def maybe_init_wandb(cfg: dict):
    exp_cfg = cfg.get("experiment", {})
    if not exp_cfg.get("use_wandb", False):
        return None

    try:
        import wandb

        wandb.init(
            project=exp_cfg.get("wandb_project", "calibration-neural-refine"),
            entity=exp_cfg.get("wandb_entity"),
            name=exp_cfg.get("name", "neural-refine"),
            config=cfg,
        )
        return wandb
    except ImportError:
        print("wandb is not installed; skipping wandb logging.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural-refine model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cpu or cuda)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint to resume from.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    set_seed(cfg.get("experiment", {}).get("seed", 1047))

    device_str = args.device or cfg.get("experiment", {}).get("device", "cpu")
    device = torch.device(device_str)

    model_type = cfg["model"].get("type", "end_to_end")
    coordinate_scale = cfg["model"].get("coordinate_scale", 100.0)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, model_type, coordinate_scale
    )

    model = build_model(cfg["model"]).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = state.get("epoch", 0) + 1
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    loss_cfg = cfg.get("loss", {})
    training_cfg = cfg["training"]
    wandb = maybe_init_wandb(cfg)
    
    # Track best model
    best_l2_px = float('inf')
    best_epoch = 0

    for epoch in range(start_epoch, training_cfg["num_epochs"] + 1):
        tic = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_cfg,
            grad_clip=training_cfg.get("grad_clip"),
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            loss_cfg,
            coordinate_scale=coordinate_scale,
        )

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = training_cfg.get("learning_rate", 1e-3)

        elapsed = time.time() - tic
        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"mae_px=({val_metrics['mae_x_px']:.2f}, {val_metrics['mae_y_px']:.2f}) | "
            f"rmse_px=({val_metrics['rmse_x_px']:.2f}, {val_metrics['rmse_y_px']:.2f}) | "
            f"l2_px={val_metrics['l2_px']:.2f} | "
            f"time={elapsed:.1f}s"
        )

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    **train_metrics,
                    **val_metrics,
                    "time_s": elapsed
                },
                step=epoch,
            )

        # Save best model based on validation l2_px
        if val_metrics['l2_px'] < best_l2_px:
            best_l2_px = val_metrics['l2_px']
            best_epoch = epoch
            ckpt_dir = resolve_path(Path(__file__).parent, training_cfg["checkpoint_dir"])
            best_path = ckpt_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, best_path)
            print(f"  → Saved best model (l2_px={best_l2_px:.2f})")

        # Regular checkpoint saving
        if epoch % training_cfg.get("checkpoint_interval", 50) == 0:
            ckpt_dir = resolve_path(Path(__file__).parent, training_cfg["checkpoint_dir"])
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    # Final test evaluation
    print(f"\nBest model was at epoch {best_epoch} with validation l2_px={best_l2_px:.2f}")
    
    # Load best model for final test evaluation
    ckpt_dir = resolve_path(Path(__file__).parent, training_cfg["checkpoint_dir"])
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded best model from {best_path}")
    
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        loss_cfg,
        coordinate_scale=coordinate_scale,
    )
    print(
        "Test metrics: "
        f"loss={test_metrics['val_loss']:.4f}, "
        f"mae_px=({test_metrics['mae_x_px']:.2f}, {test_metrics['mae_y_px']:.2f}), "
        f"rmse_px=({test_metrics['rmse_x_px']:.2f}, {test_metrics['rmse_y_px']:.2f}), "
        f"l2_px={test_metrics['l2_px']:.2f}"
    )
    if wandb is not None:
        wandb.log({
            "phase": "test",
            "best_epoch": best_epoch,
            "best_val_l2_px": best_l2_px,
            **test_metrics
        })
        wandb.finish()

    # Optionally export per-split predictions to CSV.
    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("save_predictions", False):
        base_dir = resolve_path(Path(__file__).parent, eval_cfg.get("output_dir", "../../outputs"))
        print("Saving predictions to:", base_dir)
        export_predictions(
            "train",
            train_loader.dataset,
            model,
            device,
            coordinate_scale,
            batch_size=training_cfg["batch_size"],
            num_workers=cfg["data"].get("num_workers", 0),
            output_dir=base_dir,
        )
        export_predictions(
            "val",
            val_loader.dataset,
            model,
            device,
            coordinate_scale,
            batch_size=training_cfg["batch_size"],
            num_workers=cfg["data"].get("num_workers", 0),
            output_dir=base_dir,
        )
        export_predictions(
            "test",
            test_loader.dataset,
            model,
            device,
            coordinate_scale,
            batch_size=training_cfg["batch_size"],
            num_workers=cfg["data"].get("num_workers", 0),
            output_dir=base_dir,
        )


if __name__ == "__main__":
    main()
