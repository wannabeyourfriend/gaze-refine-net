"""
Training utilities for user identification model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .model import ContrastiveLoss, GazeIdentityNet


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # Loss
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.5
    temperature: float = 0.1

    # Scheduler
    warmup_epochs: int = 5
    scheduler_type: str = "cosine"

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001


class Trainer:
    """Trainer for GazeIdentityNet."""

    def __init__(
        self,
        model: GazeIdentityNet,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=config.temperature)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Tracking
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )

        if self.config.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
                eta_min=1e-6,
            )
        else:
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
            )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_epochs],
        )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, mask, labels in dataloader:
            features = features.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, embedding = self.model(
                features, mask, return_embedding=self.config.use_contrastive_loss
            )

            # Classification loss
            loss = self.ce_loss(logits, labels)

            # Contrastive loss
            if self.config.use_contrastive_loss and embedding is not None:
                cont_loss = self.contrastive_loss(embedding, labels)
                loss = loss + self.config.contrastive_weight * cont_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, mask, labels in dataloader:
            features = features.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.model(features, mask, return_embedding=False)
            loss = self.ce_loss(logits, labels)

            total_loss += loss.item() * features.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns:
            Training history
        """
        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Check for improvement
            if val_acc > self.best_val_acc + self.config.min_delta:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                if self.checkpoint_dir:
                    self.save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1

            elapsed = time.time() - start_time

            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_acc": self.best_val_acc,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        if self.checkpoint_dir is None:
            return

        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]
