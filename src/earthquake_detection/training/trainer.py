"""
PyTorch training loop with checkpointing, early stopping, and history tracking.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Historical training and validation logs across epochs."""
    train_loss: List[float]
    val_loss: List[float]
    learning_rates: List[float]
    best_epoch: int
    best_val_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class Trainer:
    """Manages model optimization, validation, early stopping, and checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
        clip_grad_norm: Optional[float] = 1.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.clip_grad_norm = clip_grad_norm

        self.model.to(self.device)
        self.history = TrainingHistory([], [], [], 0, float("inf"))

    def train_epoch(self) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            waveforms = batch["waveform"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(waveforms)
            loss = self.criterion(preds, targets)

            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            waveforms = batch["waveform"].to(self.device)
            targets = batch["target"].to(self.device)

            preds = self.model(waveforms)
            loss = self.criterion(preds, targets)

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(self) -> TrainingHistory:
        """Execute complete training run with early stopping and checkpointing."""
        logger.info(
            f"Starting training: {self.epochs} epochs on device '{self.device}' "
            f"(Patience: {self.patience})"
        )

        epochs_no_improve = 0
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.learning_rates.append(current_lr)

            logger.info(
                f"Epoch [{epoch:03d}/{self.epochs:03d}] "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6e}"
            )

            # Checkpoint management
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.history.best_epoch = epoch
                self.history.best_val_loss = best_val_loss

                best_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path, epoch, val_loss)
                logger.info(f"[*] New best model saved to: {best_path} (Val Loss: {val_loss:.6f})")
            else:
                epochs_no_improve += 1
                if not self.save_best_only:
                    latest_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                    self.save_checkpoint(latest_path, epoch, val_loss)

            if epochs_no_improve >= self.patience:
                logger.info(
                    f"[!] Early stopping triggered after {epoch} epochs. "
                    f"Best epoch was {self.history.best_epoch} with val loss {best_val_loss:.6f}"
                )
                break

        # Save last model checkpoint
        last_path = self.checkpoint_dir / "last_model.pt"
        self.save_checkpoint(last_path, len(self.history.train_loss), self.history.val_loss[-1])

        # Save training history JSON
        history_path = self.checkpoint_dir / "training_history.json"
        self.history.save(history_path)

        return self.history

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        """Save state dictionary, optimizer state, and metadata to path."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_class": self.model.__class__.__name__,
        }
        torch.save(checkpoint, path)
