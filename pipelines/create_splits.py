"""
pipelines/create_splits.py
Create train/val/test CSV splits from a manifest file.

The manifest should have columns: video_id, video_path, label (gloss_word or integer)

Usage:
    python pipelines/create_splits.py \
        --manifest data/manifest.csv \
        --config configs/data_config.yaml
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("create_splits")


def create_splits(manifest_path: str, cfg) -> None:
    """
    Load manifest and create train/val/test splits.

    Manifest CSV columns:
        video_id         unique identifier
        video_path       path to raw .mp4
        keypoints_path   path to .npy keypoints
        label            integer class index
        gloss (optional) text transcription
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    logger.info(f"[Splits] Loaded manifest with {len(df)} videos")

    # Ensure label column exists
    if "label" not in df.columns:
        logger.error("Manifest must have a 'label' column")
        return

    # Convert string labels to integers if needed
    if df["label"].dtype == "object":
        unique_labels = sorted(df["label"].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        df["label"] = df["label"].map(label_map)

        # Save label map
        label_map_path = Path(cfg.data.splits_dir) / "label_map.json"
        Path(cfg.data.splits_dir).mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w") as f:
            json.dump(label_map, f, indent=2)
        logger.info(f"[Splits] Saved label map to {label_map_path}")

    # Create splits
    train_ratio = cfg.data.get("train_ratio", 0.75)
    val_ratio = cfg.data.get("val_ratio", 0.125)

    train_size = train_ratio
    val_size = val_ratio / (1 - train_ratio)

    train, temp = train_test_split(
        df, test_size=1 - train_size, random_state=42, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp["label"]
    )

    # Save
    splits_dir = Path(cfg.data.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = splits_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        logger.info(f"[Splits] {split_name}: {len(split_df)} samples → {path}")


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)
    create_splits(args.manifest, cfg)


if __name__ == "__main__":
    main()
