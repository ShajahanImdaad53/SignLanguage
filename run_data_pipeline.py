"""
run_data_pipeline.py
Master pipeline orchestrator — runs all data steps in order.

Steps:
    1. Process any local videos in data/raw_videos/
    2. Generate splits (or create dummy data if no videos found)
    3. Verify dataset integrity
    4. Print summary

Usage:
    python run_data_pipeline.py                      # auto-detect raw videos
    python run_data_pipeline.py --dummy              # generate dummy data
    python run_data_pipeline.py --video_dir my_vids/ # custom video directory
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("pipeline")


def print_banner(text: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


def step_process_videos(cfg, video_dir: str) -> pd.DataFrame:
    """Step 1 — extract frames + keypoints from raw videos."""
    from download_and_process_video import process_video, KeypointExtractor

    video_dir = Path(video_dir)
    video_files = list(video_dir.glob("*.mp4")) + \
                  list(video_dir.glob("*.avi")) + \
                  list(video_dir.glob("*.mov"))

    if not video_files:
        logger.warning(f"[Pipeline] No video files found in {video_dir}")
        return pd.DataFrame()

    extractor = KeypointExtractor()
    records   = []

    for vf in video_files:
        video_id = vf.stem
        logger.info(f"[Pipeline] Processing: {vf.name}")
        try:
            rec = process_video(str(vf), video_id, cfg, extractor)
            # Derive gloss from filename (e.g. "SLSL_hello_001" → "hello")
            parts = video_id.split("_")
            rec["gloss"] = parts[1] if len(parts) > 1 else video_id
            records.append(rec)
        except Exception as e:
            logger.error(f"[Pipeline] Failed for {vf.name}: {e}")

    return pd.DataFrame(records)


def step_create_dummy(cfg, num_samples: int) -> None:
    """Step 2a — generate dummy data for testing."""
    import subprocess
    cmd = [
        sys.executable, "create_test_data.py",
        "--dummy", f"--num_samples={num_samples}",
        "--config", "configs/data_config.yaml",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("create_test_data.py failed")


def step_create_splits(cfg, source_csv: str) -> None:
    """Step 2b — create train/val/test splits from existing CSV."""
    import subprocess
    cmd = [
        sys.executable, "create_test_data.py",
        f"--source={source_csv}",
        "--config", "configs/data_config.yaml",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("create_test_data.py (splits) failed")


def step_verify(cfg) -> bool:
    """Step 3 — verify all split CSVs exist and are non-empty."""
    ok = True
    for split in ["train", "val", "test"]:
        csv_path = Path(cfg.data.splits_dir) / f"{split}.csv"
        if not csv_path.exists():
            logger.error(f"[Verify] Missing: {csv_path}")
            ok = False
        else:
            df = pd.read_csv(csv_path)
            logger.info(f"[Verify] {split}: {len(df)} samples ✓")

    label_map_path = Path(cfg.data.get("label_map", "data/splits/label_map.json"))
    if not label_map_path.exists():
        logger.error(f"[Verify] Missing label_map: {label_map_path}")
        ok = False
    else:
        import json
        with open(label_map_path) as f:
            lm = json.load(f)
        logger.info(f"[Verify] label_map: {len(lm)} classes ✓")

    return ok


def step_print_summary(cfg) -> None:
    """Step 4 — print human-readable dataset summary."""
    import json

    print_banner("Dataset Summary")

    for split in ["train", "val", "test"]:
        csv_path = Path(cfg.data.splits_dir) / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {split:6s}: {len(df):>5} samples")

    lm_path = Path(cfg.data.get("label_map", "data/splits/label_map.json"))
    if lm_path.exists():
        with open(lm_path) as f:
            lm = json.load(f)
        print(f"  classes: {len(lm)}")
        print(f"  labels : {', '.join(list(lm.keys())[:10])}{'...' if len(lm) > 10 else ''}")

    print(f"\n  frames dir    : {cfg.data.frames_dir}")
    print(f"  keypoints dir : {cfg.data.keypoints_dir}")
    print(f"  splits dir    : {cfg.data.splits_dir}")
    print()


# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLT Data Pipeline Orchestrator")
    parser.add_argument("--dummy",       action="store_true",
                        help="Generate dummy data (no real videos needed)")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of dummy samples to generate")
    parser.add_argument("--video_dir",   default=None,
                        help="Path to folder of raw .mp4 files to process")
    parser.add_argument("--config",      default="configs/data_config.yaml")
    parser.add_argument("--base",        default="configs/base_config.yaml")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    cfg = load_config(args.config, base_path=args.base)

    print_banner("SLT Data Pipeline — Starting")

    # ── Step 1: Process videos ───────────────
    if args.video_dir:
        print_banner("Step 1: Processing raw videos")
        df = step_process_videos(cfg, args.video_dir)
        if not df.empty:
            interim_csv = Path(cfg.data.interim_dir) / "processed_videos.csv"
            interim_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(interim_csv, index=False)
            logger.info(f"[Pipeline] Saved {len(df)} records → {interim_csv}")
    else:
        logger.info("[Pipeline] No --video_dir provided, skipping video processing step")

    # ── Step 2: Build splits ─────────────────
    print_banner("Step 2: Building train/val/test splits")

    interim_csv = Path(cfg.data.interim_dir) / "processed_videos.csv"

    if args.dummy:
        logger.info(f"[Pipeline] Generating {args.num_samples} dummy samples")
        step_create_dummy(cfg, args.num_samples)

    elif interim_csv.exists():
        logger.info(f"[Pipeline] Building splits from {interim_csv}")
        step_create_splits(cfg, str(interim_csv))

    else:
        # Auto-detect: if raw videos exist, process them; else make dummy
        raw_dir = Path(cfg.data.raw_videos_dir)
        raw_videos = list(raw_dir.glob("*.mp4")) if raw_dir.exists() else []

        if raw_videos:
            df = step_process_videos(cfg, str(raw_dir))
            if not df.empty:
                df.to_csv(interim_csv, index=False)
                step_create_splits(cfg, str(interim_csv))
            else:
                logger.warning("[Pipeline] No processable videos. Falling back to dummy data.")
                step_create_dummy(cfg, args.num_samples)
        else:
            logger.info("[Pipeline] No videos found. Generating dummy data for testing.")
            step_create_dummy(cfg, args.num_samples)

    # ── Step 3: Verify ───────────────────────
    print_banner("Step 3: Verifying dataset integrity")
    ok = step_verify(cfg)

    if not ok:
        logger.error("[Pipeline] Verification FAILED. Check errors above.")
        sys.exit(1)

    # ── Step 4: Summary ──────────────────────
    step_print_summary(cfg)

    elapsed = time.time() - t0
    print_banner(f"Pipeline complete in {elapsed:.1f}s ✓")


if __name__ == "__main__":
    main()
