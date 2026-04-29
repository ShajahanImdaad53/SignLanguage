"""
create_test_data.py
Builds train/val/test CSV splits from processed video metadata.

Also:
  - Creates label_map.json (gloss → integer index)
  - Generates a small synthetic dataset for testing with no real videos

Usage:
    # From processed_videos.csv:
    python create_test_data.py --source data/interim/processed_videos.csv

    # Generate dummy data for quick testing (no videos needed):
    python create_test_data.py --dummy --num_samples 200
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("create_test_data")


# ─────────────────────────────────────────────
# Label map builder
# ─────────────────────────────────────────────

def build_label_map(labels: list, save_path: str) -> dict:
    """
    Create {gloss_word: integer_index} mapping.
    Saves to JSON so training and evaluation use identical mappings.
    """
    unique_labels = sorted(set(labels))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(label_map, f, indent=2)

    logger.info(f"[LabelMap] {len(label_map)} classes → {save_path}")
    return label_map


# ─────────────────────────────────────────────
# Split CSV builder
# ─────────────────────────────────────────────

def create_splits(
    df: pd.DataFrame,
    cfg,
    label_map: dict,
    splits_dir: str,
) -> None:
    """
    Split DataFrame into train/val/test CSVs.
    Stratified by label when possible.
    """
    Path(splits_dir).mkdir(parents=True, exist_ok=True)

    # Map label strings to integers
    df = df.copy()
    df["label"] = df["gloss"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    train_ratio = cfg.data.get("train_ratio", 0.75)
    val_ratio   = cfg.data.get("val_ratio",   0.125)

    # Stratified split
    try:
        train, temp = train_test_split(
            df, test_size=(1 - train_ratio),
            stratify=df["label"], random_state=42
        )
        val, test = train_test_split(
            temp, test_size=0.5,
            stratify=temp["label"], random_state=42
        )
    except ValueError:
        # Fall back to non-stratified if any class has < 2 samples
        logger.warning("[Splits] Not enough samples for stratified split. Using random split.")
        train, temp = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
        val, test   = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(f"{splits_dir}/train.csv", index=False)
    val.to_csv(f"{splits_dir}/val.csv",     index=False)
    test.to_csv(f"{splits_dir}/test.csv",   index=False)

    logger.info(f"[Splits] train={len(train)}, val={len(val)}, test={len(test)}")
    logger.info(f"[Splits] Saved to {splits_dir}/")


# ─────────────────────────────────────────────
# Dummy data generator (no real videos needed)
# ─────────────────────────────────────────────

SLSL_GLOSSES = [
    "hello", "thank_you", "please", "sorry", "yes", "no", "good", "bad",
    "help", "water", "food", "home", "school", "work", "family", "friend",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
]


def generate_dummy_dataset(
    cfg,
    num_samples: int = 200,
) -> pd.DataFrame:
    """
    Generate dummy frames and keypoints for testing without real videos.
    Creates actual .jpg frame files and .npy keypoint files on disk.

    Returns:
        DataFrame with columns: video_id, video_path, keypoints_path, gloss, label
    """
    frames_dir    = Path(cfg.data.frames_dir)
    keypoints_dir = Path(cfg.data.keypoints_dir)
    max_frames    = cfg.video.max_frames
    frame_h, frame_w = cfg.video.frame_size
    kp_dim        = cfg.model.keypoint_dim

    frames_dir.mkdir(parents=True, exist_ok=True)
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    records = []

    for i in range(num_samples):
        gloss    = SLSL_GLOSSES[i % len(SLSL_GLOSSES)]
        video_id = f"dummy_{i:04d}_{gloss}"

        # Create dummy frames directory
        video_frame_dir = frames_dir / video_id
        video_frame_dir.mkdir(parents=True, exist_ok=True)

        # Generate random coloured frames (different colour per gloss for variety)
        colour_seed = SLSL_GLOSSES.index(gloss)
        rng = np.random.RandomState(colour_seed + i)

        n_frames = rng.randint(max_frames // 2, max_frames + 1)
        for t in range(n_frames):
            # Each frame: random noise with gloss-specific colour tint
            frame = rng.randint(0, 255, (frame_h, frame_w, 3), dtype=np.uint8)
            tint  = np.array([colour_seed * 6 % 256,
                              colour_seed * 13 % 256,
                              colour_seed * 27 % 256], dtype=np.uint8)
            frame = np.clip(frame // 2 + tint, 0, 255).astype(np.uint8)
            cv2.imwrite(str(video_frame_dir / f"frame_{t:05d}.jpg"), frame)

        # Generate dummy keypoints
        kp = rng.randn(n_frames, kp_dim).astype(np.float32) * 0.1
        kp_path = keypoints_dir / f"{video_id}.npy"
        np.save(str(kp_path), kp)

        records.append({
            "video_id":       video_id,
            "video_path":     f"data/raw_videos/{video_id}.mp4",
            "keypoints_path": str(kp_path),
            "gloss":          gloss,
            "n_frames":       n_frames,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"[DummyData] Generated {i+1}/{num_samples} samples")

    df = pd.DataFrame(records)
    logger.info(f"[DummyData] Done — {len(df)} samples, {len(set(df['gloss']))} classes")
    return df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for SLT")
    parser.add_argument("--source",      help="Path to processed_videos.csv")
    parser.add_argument("--dummy",       action="store_true",
                        help="Generate synthetic dummy data for testing")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of dummy samples to generate")
    parser.add_argument("--config",      default="configs/data_config.yaml")
    parser.add_argument("--base",        default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)

    if args.dummy:
        logger.info(f"[CreateData] Generating {args.num_samples} dummy samples...")
        df = generate_dummy_dataset(cfg, num_samples=args.num_samples)
        # Save to interim
        interim_csv = Path(cfg.data.interim_dir) / "processed_videos.csv"
        interim_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(interim_csv, index=False)
        logger.info(f"[CreateData] Saved dummy metadata → {interim_csv}")

    elif args.source:
        df = pd.read_csv(args.source)
        if "gloss" not in df.columns and "label" in df.columns:
            df = df.rename(columns={"label": "gloss"})
        logger.info(f"[CreateData] Loaded {len(df)} records from {args.source}")

    else:
        parser.print_help()
        return

    # Build label map
    label_map = build_label_map(
        labels=df["gloss"].tolist(),
        save_path=cfg.data.get("label_map", "data/splits/label_map.json"),
    )

    # Assign integer labels
    df["label"] = df["gloss"].map(label_map)

    # Create splits
    create_splits(df, cfg, label_map, cfg.data.splits_dir)

    print("\n" + "="*50)
    print(f"  Dataset ready!")
    print(f"  Total samples : {len(df)}")
    print(f"  Classes       : {len(label_map)}")
    print(f"  Splits dir    : {cfg.data.splits_dir}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
