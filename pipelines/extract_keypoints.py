"""
pipelines/extract_keypoints.py
Extract MediaPipe hand + body keypoints from video frames.

Stores keypoints as .npy files (B*T, 258) where 258 = 21*3 + 21*3 + 33*3
(left hand, right hand, body pose landmarks, each with x,y,z coords)

Usage:
    python pipelines/extract_keypoints.py --video_id "my_video_001" --config configs/data_config.yaml
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError:
    print("Install mediapipe: pip install mediapipe")
    exit(1)

from PIL import Image
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("extract_keypoints")


class MediaPipeExtractor:
    """Extract hand + pose keypoints using MediaPipe."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=0.5,
        )
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=0.5,
        )

    def extract_frame(self, frame_np: np.ndarray) -> np.ndarray:
        """
        Extract keypoints from a single RGB frame.

        Args:
            frame_np: (H, W, 3) uint8 RGB image

        Returns:
            (258,) float32 array: [lh_21*3, rh_21*3, pose_33*3]
            Padded with zeros if hand/pose not detected.
        """
        keypoints = np.zeros(258, dtype=np.float32)

        # Hands (2 hands × 21 landmarks × 3 coords = 126)
        hand_result = self.mp_hands.process(frame_np)
        if hand_result.multi_hand_landmarks:
            for hand_idx, hand_lm in enumerate(hand_result.multi_hand_landmarks[:2]):
                offset = hand_idx * 63
                for lm_idx, lm in enumerate(hand_lm.landmark):
                    keypoints[offset + lm_idx * 3 : offset + lm_idx * 3 + 3] = [
                        lm.x, lm.y, lm.z
                    ]

        # Pose (33 landmarks × 3 coords = 99)
        pose_result = self.mp_pose.process(frame_np)
        if pose_result.pose_landmarks:
            pose_offset = 126
            for lm_idx, lm in enumerate(pose_result.pose_landmarks.landmark):
                keypoints[pose_offset + lm_idx * 3 : pose_offset + lm_idx * 3 + 3] = [
                    lm.x, lm.y, lm.z
                ]

        return keypoints

    def close(self):
        self.mp_hands.close()
        self.mp_pose.close()


def extract_keypoints_from_video(
    frames_dir: str,
    output_path: str,
    cfg,
) -> int:
    """
    Extract keypoints from all frames in a video directory.

    Returns:
        Number of frames processed
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        logger.error(f"[Keypoints] Frames directory not found: {frames_dir}")
        return 0

    # Load frames
    frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_files:
        logger.warning(f"[Keypoints] No frames found in {frames_dir}")
        return 0

    # Extract
    extractor = MediaPipeExtractor(
        confidence_threshold=cfg.keypoints.confidence_threshold
    )
    keypoints_list = []

    pbar = tqdm(frame_files, desc=f"Extracting keypoints")
    for frame_path in pbar:
        img = np.array(Image.open(frame_path).convert("RGB"))
        kp = extractor.extract_frame(img)
        keypoints_list.append(kp)

    extractor.close()

    # Stack and save
    keypoints_array = np.stack(keypoints_list)  # (T, 258)
    np.save(str(output_path), keypoints_array)

    logger.info(f"[Keypoints] Saved {len(keypoints_list)} frames to {output_path}")
    return len(keypoints_list)


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints")
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)

    frames_dir = Path(cfg.data.frames_dir) / args.video_id
    output_path = Path(cfg.data.keypoints_dir) / f"{args.video_id}.npy"

    count = extract_keypoints_from_video(str(frames_dir), str(output_path), cfg)
    logger.info(f"[Pipeline] Keypoint extraction complete: {count} frames")


if __name__ == "__main__":
    main()
