"""
src/trainer.py
Supervised Fine-Tuning Trainer for Sign Language Translation.

Loads SSL-pretrained weights, then trains end-to-end on labelled data.

Run with:
    python src/trainer.py --config configs/train_config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.models.slt_model import build_model
from src.dataset import build_dataloaders
from src.evaluate import evaluate
from src.utils.config import load_config, save_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("trainer")


class Trainer:
    """Manages fine-tuning of the SLT model on labelled sign data."""

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Trainer] Device: {self.device}")

        # ── Model ────────────────────────────────────────────
        self.model = build_model(cfg).to(self.device)

        # Load SSL pretrained weights if available
        ssl_checkpoint = cfg.ssl.pretraining.save_path
        if Path(ssl_checkpoint).exists():
            state = torch.load(ssl_checkpoint, map_location=self.device)
            # Load only matching keys (skip classifier, ssl_decoder)
            model_state = self.model.state_dict()
            matched = {k: v for k, v in state.items()
                       if k in model_state and v.shape == model_state[k].shape}
            model_state.update(matched)
            self.model.load_state_dict(model_state)
            logger.info(f"[Trainer] Loaded SSL weights: {ssl_checkpoint} "
                        f"({len(matched)}/{len(model_state)} layers matched)")
        else:
            logger.warning("[Trainer] No SSL checkpoint found. Training from scratch.")

        # Unfreeze backbone for end-to-end fine-tuning
        self.model.backbone.unfreeze()

        # ── Optimizer ────────────────────────────────────────
        t = cfg.training
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=t.learning_rate,
            weight_decay=t.weight_decay,
        )

        # Warmup + cosine decay
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=t.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=t.epochs - t.warmup_epochs,
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[t.warmup_epochs]
        )

        self.scaler = GradScaler(enabled=t.get("mixed_precision", True))

        # ── Loss ─────────────────────────────────────────────
        label_smoothing = cfg.get("loss", {}).get("label_smoothing", 0.1) \
                          if hasattr(cfg, "get") else 0.1
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ── Data ─────────────────────────────────────────────
        self.loaders = build_dataloaders(cfg)
        self.train_loader = self.loaders.get("train")
        self.val_loader   = self.loaders.get("val")

        # ── Config shortcuts ─────────────────────────────────
        self.epochs     = t.epochs
        self.grad_clip  = t.gradient_clip
        self.patience   = t.early_stopping_patience
        self.save_dir   = Path(cfg.logging.save_dir)
        self.log_dir    = Path(cfg.logging.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ─────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}/{self.epochs}")
        for frames, keypoints, labels in pbar:
            frames    = frames.to(self.device)
            keypoints = keypoints.to(self.device)
            labels    = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=True):
                out    = self.model(frames, keypoints, mode="finetune")
                logits = out["logits"]           # (B, vocab_size)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n_batches, 1)

    # ─────────────────────────────────────────────────────────
    # Validation step
    # ─────────────────────────────────────────────────────────

    def _val_epoch(self) -> dict:
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_acc": 0.0}
        return evaluate(self.model, self.val_loader, self.criterion, self.device)

    # ─────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────

    def train(self) -> None:
        if self.train_loader is None:
            logger.error("[Trainer] No training data. Aborting.")
            return

        logger.info(f"[Trainer] Starting fine-tuning for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._val_epoch()
            self.scheduler.step()

            val_loss = val_metrics.get("val_loss", 0.0)
            val_acc  = val_metrics.get("val_acc", 0.0)
            lr       = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"[Trainer] Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.3f} | "
                f"lr={lr:.2e}"
            )

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                ckpt_path = self.save_dir / "best_model.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"[Trainer] New best val_acc={val_acc:.3f} → {ckpt_path}")
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if epoch % self.cfg.logging.checkpoint_every == 0:
                ckpt_path = self.save_dir / f"checkpoint_epoch{epoch:03d}.pt"
                torch.save(self.model.state_dict(), ckpt_path)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"[Trainer] Early stopping at epoch {epoch} "
                            f"(no improvement for {self.patience} epochs)")
                break

        # Save training history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"[Trainer] Training history saved to {history_path}")
        logger.info(f"[Trainer] Best validation accuracy: {self.best_val_acc:.3f}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Trainer for SLT")
    parser.add_argument("--config",     default="configs/train_config.yaml")
    parser.add_argument("--base",       default="configs/base_config.yaml")
    parser.add_argument("--ssl_config", default="configs/ssl_config.yaml")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # Merge all configs
    cfg = load_config(args.config, base_path=args.base)
    ssl_cfg = load_config(args.ssl_config, base_path=args.base)
    cfg["ssl"] = ssl_cfg.get("ssl", {})

    set_seed(args.seed)
    save_config(cfg, "logs/train_config_snapshot.yaml")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
