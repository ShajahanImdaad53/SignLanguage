"""
download_and_process_video.py
Download and extract frames + keypoints from sign language videos.
"""

import argparse
import os
import cv2
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("download_and_process")


def download_video(url: str, output_path: str) -> bool:
    """Download video using yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        logger.error("[Download] yt_dlp not installed")
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "best[ext=mp4]",
        "outtmpl": str(output_path).replace(".mp4", ""),
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"[Download] Fetching {url}...")
            ydl.download([url])
        return True
    except Exception as e:
        logger.error(f"[Download] Failed: {e}")
        return False


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 25,
    max_frames: Optional[int] = None,
) -> int:
    """Extract frames from video at specified FPS."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(video_fps / fps) if fps > 0 else 1

    frame_count = 0
    frame_idx = 0

    logger.info(f"[Extract] Processing {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame = cv2.resize(frame, (224, 224))
            output_path = Path(output_dir) / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info(f"[Extract] Extracted {frame_count} frames")
    return frame_count


def extract_keypoints_mediapipe(
    frames_dir: str,
    output_path: str,
) -> bool:
    """Extract MediaPipe keypoints from frames."""
    try:
        import mediapipe as mp
        import numpy as np
    except ImportError:
        logger.error("[Keypoints] mediapipe not installed")
        return False

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    keypoints_list = []
    frames = sorted(Path(frames_dir).glob("*.jpg"))

    logger.info(f"[Keypoints] Extracting from {len(frames)} frames")

    pose_detector = mp_pose.Pose(static_image_mode=False)
    hands_detector = mp_hands.Hands(static_image_mode=False)

    for frame_idx, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kp_vec = []

        # Pose
        result_pose = pose_detector.process(image_rgb)
        if result_pose.pose_landmarks:
            for lm in result_pose.pose_landmarks.landmark:
                kp_vec.extend([lm.x, lm.y, lm.z])
        else:
            kp_vec.extend([0.0] * 99)

        # Hands
        result_hands = hands_detector.process(image_rgb)
        if result_hands.multi_hand_landmarks:
            for hand_lms in result_hands.multi_hand_landmarks[:2]:
                for lm in hand_lms.landmark:
                    kp_vec.extend([lm.x, lm.y, lm.z])
        while len(kp_vec) < 258:
            kp_vec.append(0.0)

        keypoints_list.append(kp_vec[:258])

    keypoints_array = np.array(keypoints_list, dtype=np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, keypoints_array)

    logger.info(f"[Keypoints] Saved to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Video URL")
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--local_path", help="Local video file")
    parser.add_argument("--output_dir", default="data/raw_videos/")
    parser.add_argument("--frames_dir", default="data/interim/frames/")
    parser.add_argument("--keypoints_dir", default="data/interim/keypoints/")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--extract_keypoints", action="store_true")

    args = parser.parse_args()

    if args.local_path:
        video_path = args.local_path
    elif args.url:
        video_path = os.path.join(args.output_dir, f"{args.video_id}.mp4")
        if not download_video(args.url, video_path):
            return
    else:
        logger.error("Either --url or --local_path required")
        return

    video_frames_dir = os.path.join(args.frames_dir, args.video_id)
    extract_frames(video_path, video_frames_dir, fps=args.fps)

    if args.extract_keypoints:
        keypoints_path = os.path.join(args.keypoints_dir, f"{args.video_id}.npy")
        extract_keypoints_mediapipe(video_frames_dir, keypoints_path)

    logger.info(f"[Main] Complete: {args.video_id}")


if __name__ == "__main__":
    main()
