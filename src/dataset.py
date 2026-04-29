"""
src/dataset.py
PyTorch Dataset for Sign Language Translation.

Each row in train/val/test.csv:
    video_id, video_path, keypoints_path, label, gloss_sequence

Returns (frames_tensor, keypoints_tensor, label_index) per sample.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict

from src.utils.logger import get_logger

logger = get_logger("dataset")


# ─────────────────────────────────────────────
# Augmentation transforms
# ─────────────────────────────────────────────

def build_transforms(cfg, split: str = "train") -> transforms.Compose:
    """Return torchvision transforms for frames based on split."""
    mean = cfg.video.normalize_mean
    std  = cfg.video.normalize_std
    size = cfg.video.frame_size  # [H, W]

    if split == "train":
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ─────────────────────────────────────────────
# Frame loading helper
# ─────────────────────────────────────────────

def load_frames(frames_dir: str, video_id: str, max_frames: int) -> np.ndarray:
    """
    Load pre-extracted frame images from disk.
    Falls back to zeros if directory missing (for testing).

    Args:
        frames_dir: Root directory containing per-video frame folders
        video_id:   Sub-folder name matching the video
        max_frames: How many frames to return (pad/truncate)

    Returns:
        np.ndarray of shape (max_frames, H, W, 3) — uint8
    """
    video_frames_dir = Path(frames_dir) / video_id
    frames = []

    if video_frames_dir.exists():
        frame_files = sorted(video_frames_dir.glob("*.jpg")) + \
                      sorted(video_frames_dir.glob("*.png"))
        for fp in frame_files[:max_frames]:
            img = Image.open(fp).convert("RGB")
            frames.append(np.array(img))

    # Pad with zeros if fewer frames than max_frames
    if len(frames) == 0:
        logger.warning(f"No frames found for video_id={video_id}, using zeros")
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * max_frames
    elif len(frames) < max_frames:
        pad = [frames[-1]] * (max_frames - len(frames))
        frames.extend(pad)

    return np.stack(frames[:max_frames])  # (T, H, W, 3)


def load_keypoints(keypoints_path: str, max_frames: int, keypoint_dim: int) -> np.ndarray:
    """
    Load pre-extracted MediaPipe keypoints from .npy file.

    Returns:
        np.ndarray of shape (max_frames, keypoint_dim) — float32
    """
    kp_path = Path(keypoints_path)
    if kp_path.exists():
        kp = np.load(str(kp_path)).astype(np.float32)   # (T, D)
        T = kp.shape[0]
        if T < max_frames:
            pad = np.zeros((max_frames - T, keypoint_dim), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        return kp[:max_frames]
    else:
        logger.warning(f"Keypoints not found: {keypoints_path}, using zeros")
        return np.zeros((max_frames, keypoint_dim), dtype=np.float32)


# ─────────────────────────────────────────────
# Main Dataset
# ─────────────────────────────────────────────

class SLTDataset(Dataset):
    """
    Sign Language Translation Dataset.

    CSV columns expected:
        video_id        unique identifier (used to find frame folder)
        video_path      path to raw mp4 (informational)
        keypoints_path  path to .npy keypoints file
        label           integer class index
        gloss           text gloss string (optional, for seq2seq)
    """

    def __init__(
        self,
        csv_path: str,
        cfg,
        split: str = "train",
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.split = split
        self.label_map = label_map or {}
        self.max_frames = cfg.video.max_frames
        self.keypoint_dim = cfg.model.keypoint_dim
        self.frames_dir = cfg.data.frames_dir
        self.transform = build_transforms(cfg, split)

        logger.info(f"[Dataset] {split}: {len(self.df)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])

        # ── Frames ──────────────────────────────
        raw_frames = load_frames(self.frames_dir, video_id, self.max_frames)
        # Apply transform to each frame individually
        frame_tensors = []
        for i in range(self.max_frames):
            img = Image.fromarray(raw_frames[i])
            frame_tensors.append(self.transform(img))
        frames = torch.stack(frame_tensors)   # (T, C, H, W)

        # ── Keypoints ────────────────────────────
        kp_path = str(row.get("keypoints_path", ""))
        keypoints = load_keypoints(kp_path, self.max_frames, self.keypoint_dim)
        keypoints = torch.from_numpy(keypoints)  # (T, D)

        # ── Label ────────────────────────────────
        label = int(row.get("label", 0))

        return frames, keypoints, label


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def build_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Build train, val, and test DataLoaders from config.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    splits = {
        "train": cfg.data.train_csv,
        "val":   cfg.data.val_csv,
        "test":  cfg.data.test_csv,
    }

    label_map = {}
    label_map_path = Path(cfg.data.get("label_map", ""))
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)

    loaders = {}
    for split, csv_path in splits.items():
        if not Path(csv_path).exists():
            logger.warning(f"[DataLoader] CSV not found for {split}: {csv_path}, skipping")
            continue

        dataset = SLTDataset(csv_path, cfg, split=split, label_map=label_map)
        shuffle = (split == "train")

        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.get("num_workers", 4),
            pin_memory=cfg.training.get("pin_memory", True),
            drop_last=(split == "train"),
        )
        logger.info(f"[DataLoader] {split}: {len(dataset)} samples, "
                    f"batch={cfg.training.batch_size}, "
                    f"batches/epoch={len(loaders[split])}")

    return loaders
