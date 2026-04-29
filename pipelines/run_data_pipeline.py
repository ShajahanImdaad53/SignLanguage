"""
pipelines/run_data_pipeline.py
Master pipeline — orchestrates download, frame extraction, keypoint extraction.

This is the entry point for data preparation.

Usage:
    python pipelines/run_data_pipeline.py \
        --manifest data/manifest.csv \
        --config configs/data_config.yaml
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from pipelines.download_and_process_video import process_video
from pipelines.extract_keypoints import extract_keypoints_from_video
from pipelines.create_splits import create_splits
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("data_pipeline")


def run_full_pipeline(manifest_path: str, cfg, max_workers: int = 4) -> None:
    """
    Full data pipeline:
        1. Download videos (parallel)
        2. Extract frames (parallel)
        3. Extract keypoints (parallel)
        4. Create train/val/test splits
        5. Save manifest with paths
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    logger.info(f"[Pipeline] Starting with {len(df)} videos from {manifest_path}")

    # ─────────────────────────────────────────
    # Step 1: Download & extract frames
    # ─────────────────────────────────────────
    logger.info("[Pipeline] Step 1: Download videos and extract frames...")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            url = str(row["video_path"])  # Interpret as URL/path
            future = executor.submit(process_video, url, video_id, cfg)
            futures[future] = video_id

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Download & Frame Extract"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed for {futures[future]}: {e}")

    successful = [r for r in results if r.get("success")]
    logger.info(f"[Pipeline] Downloaded & extracted: {len(successful)}/{len(df)}")

    # ─────────────────────────────────────────
    # Step 2: Extract keypoints
    # ─────────────────────────────────────────
    logger.info("[Pipeline] Step 2: Extract MediaPipe keypoints...")

    keypoint_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for result in successful:
            video_id = result["video_id"]
            frames_dir = result["frames_dir"]
            output_path = Path(cfg.data.keypoints_dir) / f"{video_id}.npy"
            future = executor.submit(
                extract_keypoints_from_video, frames_dir, str(output_path), cfg
            )
            futures[future] = video_id

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Keypoint Extract"):
            try:
                num_frames = future.result()
                keypoint_results.append((futures[future], num_frames))
            except Exception as e:
                logger.error(f"Keypoint extraction failed for {futures[future]}: {e}")

    # ─────────────────────────────────────────
    # Step 3: Create splits
    # ─────────────────────────────────────────
    logger.info("[Pipeline] Step 3: Creating train/val/test splits...")
    create_splits(str(manifest_path), cfg)

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    logger.info("[Pipeline] ✓ COMPLETE")
    logger.info(f"  Downloaded:        {len(successful)}/{len(df)} videos")
    logger.info(f"  Keypoints:         {len(keypoint_results)}/{len(successful)}")
    logger.info(f"  Frames dir:        {cfg.data.frames_dir}")
    logger.info(f"  Keypoints dir:     {cfg.data.keypoints_dir}")
    logger.info(f"  Splits dir:        {cfg.data.splits_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full data preparation pipeline")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)
    run_full_pipeline(args.manifest, cfg, max_workers=args.workers)


if __name__ == "__main__":
    main()
