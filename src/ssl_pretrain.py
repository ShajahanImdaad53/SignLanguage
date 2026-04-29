"""
src/ssl_pretrain.py
Self-Supervised Pretraining for the SLT model.

Two SSL strategies:
    1. Masked Autoencoder (MAE) — mask frames, reconstruct visual features
    2. Contrastive Learning (SimCLR-style) — learn similar representations
       for two augmented views of the same clip

Run with:
    python src/ssl_pretrain.py --config configs/ssl_config.yaml
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.models.slt_model import build_model
from src.dataset import build_dataloaders
from src.utils.config import load_config, save_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("ssl_pretrain")


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def masked_reconstruction_loss(
    reconstructed: torch.Tensor,
    original_features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss computed only on masked (hidden) positions.

    Args:
        reconstructed:     (B, T, D) — model's reconstruction
        original_features: (B, T, D) — target (CNN features before masking)
        mask:              (B, T) bool — True = masked position

    Returns:
        Scalar loss
    """
    # Only compute loss where mask is True
    loss = F.mse_loss(reconstructed[mask], original_features[mask], reduction="mean")
    return loss


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    NT-Xent (SimCLR) contrastive loss.
    z1 and z2 are embeddings of two augmented views of the same clips.

    Args:
        z1, z2:      (B, D) — L2-normalised embeddings
        temperature: Softmax temperature (lower = sharper)

    Returns:
        Scalar loss
    """
    B = z1.shape[0]

    # L2 normalise
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Concatenate: (2B, D)
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix: (2B, 2B)
    sim = torch.mm(z, z.t()) / temperature

    # Mask out self-similarity on diagonal
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


# ─────────────────────────────────────────────
# Masking
# ─────────────────────────────────────────────

def random_temporal_mask(
    B: int, T: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """
    Create random temporal mask.

    Returns:
        (B, T) bool tensor — True = masked (hidden from model)
    """
    num_masked = int(T * mask_ratio)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(T, device=device)[:num_masked]
        mask[b, idx] = True
    return mask


def block_temporal_mask(
    B: int, T: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """
    Contiguous block masking — mask a random contiguous span.
    Harder than random; forces the model to predict across time gaps.
    """
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    num_masked = int(T * mask_ratio)
    for b in range(B):
        start = torch.randint(0, T - num_masked + 1, (1,)).item()
        mask[b, start : start + num_masked] = True
    return mask


# ─────────────────────────────────────────────
# Pretrainer
# ─────────────────────────────────────────────

class SSLPretrainer:
    """Manages the full SSL pretraining loop."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[SSL] Device: {self.device}")

        # Model
        self.model = build_model(cfg).to(self.device)

        # Freeze backbone during SSL (train transformer + decoder only)
        self.model.backbone.freeze()

        # Optimizer
        ssl = cfg.ssl
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=ssl.pretraining.lr,
            weight_decay=ssl.pretraining.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=ssl.pretraining.epochs,
        )
        self.scaler = GradScaler(enabled=cfg.training.get("mixed_precision", True))

        # Data
        self.loaders = build_dataloaders(cfg)
        self.train_loader = self.loaders.get("train")

        # Config shortcuts
        self.method       = ssl.method
        self.mask_ratio   = ssl.masking_ratio
        self.mask_strategy = ssl.get("mask_strategy", "random")
        self.temperature  = ssl.contrastive.temperature if hasattr(ssl, "contrastive") else 0.07
        self.epochs       = ssl.pretraining.epochs
        self.save_path    = ssl.pretraining.save_path

        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

    def _make_mask(self, B: int, T: int) -> torch.Tensor:
        if self.mask_strategy == "block":
            return block_temporal_mask(B, T, self.mask_ratio, self.device)
        return random_temporal_mask(B, T, self.mask_ratio, self.device)

    def _step_mae(self, frames: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """One MAE training step."""
        B, T = frames.shape[:2]

        # Get clean visual features (target for reconstruction)
        with torch.no_grad():
            target_features = self.model.backbone(frames)  # (B, T, D)

        # Create mask
        mask = self._make_mask(B, T)

        # Forward with masking
        with autocast(enabled=self.cfg.training.get("mixed_precision", True)):
            out = self.model(frames, keypoints, mask=mask, mode="pretrain")
            loss = masked_reconstruction_loss(
                out["reconstructed"], target_features, mask
            )

        return loss

    def _step_contrastive(
        self,
        frames: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> torch.Tensor:
        """One SimCLR contrastive step — two views of the same batch."""
        # View 1: original
        # View 2: apply temporal jitter (shift frames by small offset)
        B, T, C, H, W = frames.shape
        shift = max(1, int(T * 0.1))
        frames2 = torch.roll(frames, shifts=shift, dims=1)

        with autocast(enabled=self.cfg.training.get("mixed_precision", True)):
            enc1 = self.model.encode(frames, keypoints)     # (B, T, D)
            enc2 = self.model.encode(frames2, keypoints)

            # Pool over time for contrastive
            z1 = enc1.mean(dim=1)   # (B, D)
            z2 = enc2.mean(dim=1)

            loss = contrastive_loss(z1, z2, self.temperature)

        return loss

    def train(self) -> None:
        """Full SSL pretraining loop."""
        if self.train_loader is None:
            logger.error("[SSL] No training data available. Aborting.")
            return

        logger.info(f"[SSL] Starting {self.method} pretraining for {self.epochs} epochs")
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches  = 0

            pbar = tqdm(self.train_loader, desc=f"SSL Epoch {epoch}/{self.epochs}")
            for frames, keypoints, _ in pbar:
                frames    = frames.to(self.device)
                keypoints = keypoints.to(self.device)

                self.optimizer.zero_grad()

                # Choose SSL strategy
                if self.method == "masked_autoencoder":
                    loss = self._step_mae(frames, keypoints)
                elif self.method in ("simclr", "contrastive"):
                    loss = self._step_contrastive(frames, keypoints)
                else:
                    raise ValueError(f"Unknown SSL method: {self.method}")

                # Backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_batches  += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()

            avg_loss = total_loss / max(n_batches, 1)
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"[SSL] Epoch {epoch:03d} | loss={avg_loss:.4f} | lr={lr:.2e}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"[SSL] Saved best model → {self.save_path}")

        logger.info(f"[SSL] Pretraining complete. Best loss: {best_loss:.4f}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining for SLT")
    parser.add_argument("--config", default="configs/ssl_config.yaml")
    parser.add_argument("--base",   default="configs/base_config.yaml")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)
    set_seed(args.seed)

    # Save config snapshot for reproducibility
    save_config(cfg, f"logs/ssl_config_snapshot.yaml")

    pretrainer = SSLPretrainer(cfg)
    pretrainer.train()


if __name__ == "__main__":
    main()
