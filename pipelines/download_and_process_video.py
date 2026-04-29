"""
pipelines/download_and_process_video.py
Download raw sign language videos and extract frames.

Usage:
    python pipelines/download_and_process_video.py \
        --url "https://youtube.com/watch?v=..." \
        --video_id "my_video_001" \
        --config configs/data_config.yaml
"""

import argparse
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("download_video")


def download_video(url: str, output_path: str) -> bool:
    """
    Download video from URL using yt-dlp.
    Falls back to manual download if URL is local file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if it's a local file
    if os.path.exists(url):
        logger.info(f"[Video] Local file detected: {url}")
        import shutil
        shutil.copy(url, output_path)
        return True

    # Download from URL
    try:
        cmd = [
            "yt-dlp",
            "-f", "best",
            "-o", str(output_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"[Video] Downloaded: {output_path}")
            return True
        else:
            logger.error(f"[Video] Download failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"[Video] Download error: {e}")
        return False


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 25,
    frame_size: tuple = (224, 224),
    max_frames: int = 64,
) -> int:
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to .mp4 file
        output_dir: Where to save frame .jpg files
        fps: Sample rate (25 fps typical for sign language)
        frame_size: Resize to (H, W)
        max_frames: Stop after this many frames

    Returns:
        Number of frames extracted
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[Frame] Failed to open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    count = 0
    frame_num = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample at desired FPS
        if frame_num % frame_interval == 0:
            # Resize
            frame = cv2.resize(frame, frame_size)
            # Convert BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save
            out_path = output_dir / f"frame_{count:06d}.jpg"
            Image.fromarray(frame).save(out_path)
            count += 1

        frame_num += 1

    cap.release()
    logger.info(f"[Frame] Extracted {count} frames from {video_path}")
    return count


def process_video(
    url: str,
    video_id: str,
    cfg,
) -> dict:
    """
    Full pipeline: download + extract frames.

    Returns:
        dict with metadata (path, num_frames, etc.)
    """
    # Download
    raw_video_dir = Path(cfg.data.raw_videos_dir)
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    raw_video_path = raw_video_dir / f"{video_id}.mp4"

    if not raw_video_path.exists():
        success = download_video(url, str(raw_video_path))
        if not success:
            return {"success": False, "video_id": video_id}
    else:
        logger.info(f"[Video] Already downloaded: {raw_video_path}")

    # Extract frames
    frames_dir = Path(cfg.data.frames_dir) / video_id
    num_frames = extract_frames(
        str(raw_video_path),
        str(frames_dir),
        fps=cfg.video.fps,
        frame_size=tuple(cfg.video.frame_size),
        max_frames=cfg.video.max_frames,
    )

    return {
        "success": True,
        "video_id": video_id,
        "video_path": str(raw_video_path),
        "frames_dir": str(frames_dir),
        "num_frames": num_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Download and extract video frames")
    parser.add_argument("--url", required=True, help="Video URL or local path")
    parser.add_argument("--video_id", required=True, help="Unique video identifier")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)
    result = process_video(args.url, args.video_id, cfg)

    if result["success"]:
        logger.info(f"[Pipeline] SUCCESS: {result}")
    else:
        logger.error(f"[Pipeline] FAILED: {result}")


if __name__ == "__main__":
    main()
