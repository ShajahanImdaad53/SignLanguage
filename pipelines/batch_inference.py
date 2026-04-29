"""
pipelines/batch_inference.py
Batch inference on multiple sign language videos.

Usage:
    python pipelines/batch_inference.py \
        --input_dir data/raw_videos/ \
        --checkpoint models/best_model.pt \
        --output results.json
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

from src.inference import SignLanguageTranslator
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("batch_inference")


def process_single_video(video_path: str, translator: SignLanguageTranslator) -> dict:
    """Process a single video."""
    try:
        result = translator.translate_video(str(video_path))
        result["video_path"] = str(video_path)
        result["status"] = "success"
        return result
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        return {
            "video_path": str(video_path),
            "status": "error",
            "error": str(e),
        }


def batch_inference(
    input_dir: str,
    translator: SignLanguageTranslator,
    output_path: str = "results.json",
    max_workers: int = 4,
) -> None:
    """
    Run inference on all videos in a directory.

    Args:
        input_dir: Directory containing .mp4 files
        translator: SignLanguageTranslator instance
        output_path: Where to save results JSON
        max_workers: Number of parallel workers
    """
    input_dir = Path(input_dir)
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mov"))

    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return

    logger.info(f"[Batch] Found {len(video_files)} videos")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, str(vf), translator): vf
            for vf in video_files
        }

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Batch Inference")
        for future in pbar:
            try:
                result = future.result()
                results.append(result)
                pbar.set_postfix({"status": result.get("status", "unknown")})
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]

    logger.info(f"[Batch] ✓ COMPLETE")
    logger.info(f"  Successful: {len(successful)}/{len(results)}")
    logger.info(f"  Failed:     {len(failed)}/{len(results)}")
    logger.info(f"  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch inference on videos")
    parser.add_argument("--input_dir", default="data/raw_videos/")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label_map", default="data/splits/label_map.json")
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)
    translator = SignLanguageTranslator(
        args.checkpoint, cfg, label_map_path=args.label_map
    )

    batch_inference(
        args.input_dir,
        translator,
        output_path=args.output,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
