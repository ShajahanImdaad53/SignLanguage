"""
src/inference.py
Inference module — predict sign glosses from new videos.
"""

import json
from pathlib import Path

import torch
import numpy as np
import cv2

from src.models.slt_model import SLTModel
from src.utils.config import load_config
from src.utils.logger import get_logger
from pipelines.extract_keypoints import MediaPipeExtractor

logger = get_logger("inference")


class SignLanguageTranslator:
    """Single-video inference pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        base_config_path: str = "configs/base_config.yaml",
        data_config_path: str = "configs/data_config.yaml",
    ):
        """Initialize translator."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config(data_config_path, base_path=base_config_path)

        # Load model
        self.model = SLTModel(self.cfg).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(f"[Translator] Loaded model from {checkpoint_path}")

        # Load label map
        label_map_path = Path(self.cfg.data.get("label_map", "data/splits/label_map.json"))
        self.reverse_label_map = {}
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_map = json.load(f)
                self.reverse_label_map = {v: k for k, v in label_map.items()}

        # MediaPipe extractor
        self.kp_extractor = MediaPipeExtractor(
            confidence_threshold=self.cfg.keypoints.confidence_threshold
        )
        self.max_frames = self.cfg.video.max_frames
        self.frame_size = tuple(self.cfg.video.frame_size)

    def predict_from_video(self, video_path: str) -> dict:
        """Predict sign gloss from video file."""
        logger.info(f"[Translator] Processing: {video_path}")

        frames = self._extract_frames(video_path)
        if frames is None:
            return {"error": "Failed to extract frames"}

        keypoints = self._extract_keypoints(frames)
        logits, confidence = self._run_inference(frames, keypoints)

        pred_idx = logits.argmax().item()
        pred_gloss = self.reverse_label_map.get(pred_idx, f"class_{pred_idx}")

        return {
            "gloss": pred_gloss,
            "confidence": float(confidence),
            "pred_idx": int(pred_idx),
        }

    def _extract_frames(self, video_path: str) -> np.ndarray:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open: {video_path}")
            return None

        frames = []
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cv2.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            return None

        if len(frames) < self.max_frames:
            pad = [frames[-1]] * (self.max_frames - len(frames))
            frames.extend(pad)

        return np.stack(frames[:self.max_frames])

    def _extract_keypoints(self, frames: np.ndarray) -> np.ndarray:
        """Extract keypoints from frames."""
        keypoints_list = []
        for frame in frames:
            kp = self.kp_extractor.extract_frame(frame)
            keypoints_list.append(kp)
        return np.stack(keypoints_list)

    def _run_inference(self, frames: np.ndarray, keypoints: np.ndarray) -> tuple:
        """Run model inference."""
        frames_tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 255.0
        frames_tensor = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)
        keypoints_tensor = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(frames_tensor, keypoints_tensor, mode="finetune")
            logits = out["logits"]
            confidence = torch.softmax(logits, dim=1).max().item()

        return logits[0], confidence

    def close(self):
        """Cleanup."""
        self.kp_extractor.close()
