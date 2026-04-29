"""
agents/training_agent.py
Autonomous Training Agent — monitors experiments, logs metrics, triggers callbacks.
"""

import json
from pathlib import Path
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger("training_agent")


class TrainingAgent:
    """Manages training lifecycle: checkpointing, early stopping."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = Path(cfg.logging.save_dir)
        self.log_dir = Path(cfg.logging.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.patience = cfg.training.early_stopping_patience
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

        self.use_wandb = cfg.logging.get("wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="slt-hybrid-ssl",
                    name=cfg.logging.get("experiment_name", "experiment"),
                )
                logger.info("[Agent] W&B initialized")
            except ImportError:
                self.use_wandb = False

    def on_epoch_start(self, epoch: int) -> None:
        """Called at start of each epoch."""
        logger.info(f"[Agent] Epoch {epoch} starting")

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float,
                     val_acc: float, lr: float = 0.0) -> Dict:
        """Called at end of epoch. Returns early stopping signal."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(lr)

        if self.use_wandb:
            try:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "val_acc": val_acc, "lr": lr})
            except:
                pass

        improved = val_acc > self.best_val_acc
        if improved:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            return {"should_stop": False, "improved": True}
        else:
            self.patience_counter += 1
            should_stop = self.patience_counter >= self.patience
            return {"should_stop": should_stop, "improved": False}

    def on_training_end(self) -> None:
        """Called when training finishes."""
        hist_path = self.log_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"[Agent] Best val_acc: {self.best_val_acc:.3f}")
