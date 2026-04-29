"""
pipelines/inference_pipeline.py
Production inference — takes a video file and outputs predicted sign gloss.

Usage:
    python pipelines/inference_pipeline.py --video data/raw_videos/my_sign.mp4
    python pipelines/inference_pipeline.py --video my_sign.mp4 --top_k 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from download_and_process_video import extract_frames, KeypointExtractor
from src.models.slt_model import build_model
from src.dataset import build_transforms, load_keypoints
from src.utils.config import load_config
from src.utils.logger import get_logger
from PIL import Image

logger = get_logger("inference")


class SLTInferencer:
    """
    Single-video sign language translation inference.
    Loads a trained model and label map, returns top-k gloss predictions.
    """

    def __init__(self, cfg, checkpoint_path: str, label_map_path: str):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = build_model(cfg).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(f"[Inference] Loaded model from {checkpoint_path}")

        # Load label map (reverse: index → gloss)
        with open(label_map_path) as f:
            lm = json.load(f)
        self.idx_to_label = {v: k for k, v in lm.items()}

        # Transforms + keypoint extractor
        self.transform = build_transforms(cfg, split="val")
        self.extractor = KeypointExtractor()
        self.kp_dim    = cfg.model.keypoint_dim
        self.max_frames = cfg.video.max_frames

    @torch.no_grad()
    def predict(self, video_path: str, top_k: int = 5) -> list:
        """
        Predict sign gloss from a video file.

        Args:
            video_path: Path to .mp4 video
            top_k:      Number of top predictions to return

        Returns:
            List of (gloss, confidence) tuples sorted by confidence desc
        """
        video_path = Path(video_path)
        video_id   = video_path.stem
        tmp_frames = Path("data/interim/frames_tmp")

        # Extract frames
        extract_frames(
            video_path=str(video_path),
            output_dir=str(tmp_frames),
            video_id=video_id,
            fps=self.cfg.video.fps,
            frame_size=tuple(self.cfg.video.frame_size),
            max_frames=self.max_frames,
        )

        # Load + transform frames
        frame_dir = tmp_frames / video_id
        frame_files = sorted(frame_dir.glob("*.jpg"))
        frames = []
        for fp in frame_files[:self.max_frames]:
            img = Image.open(fp).convert("RGB")
            frames.append(self.transform(img))

        # Pad if needed
        while len(frames) < self.max_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, *self.cfg.video.frame_size))

        frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        # Extract + load keypoints
        kp_path = tmp_frames / f"{video_id}.npy"
        self.extractor.extract_video(
            frames_dir=str(tmp_frames),
            video_id=video_id,
            output_path=str(kp_path),
        )
        kp = load_keypoints(str(kp_path), self.max_frames, self.kp_dim)
        kp_tensor = torch.from_numpy(kp).unsqueeze(0).to(self.device)  # (1, T, D)

        # Forward pass
        out    = self.model(frames_tensor, kp_tensor, mode="finetune")
        logits = out["logits"][0]           # (vocab_size,)
        probs  = F.softmax(logits, dim=-1)

        k      = min(top_k, probs.shape[0])
        top_probs, top_idx = probs.topk(k)

        predictions = [
            (self.idx_to_label.get(idx.item(), f"class_{idx.item()}"),
             round(prob.item(), 4))
            for prob, idx in zip(top_probs, top_idx)
        ]

        return predictions


def main():
    parser = argparse.ArgumentParser(description="SLT Inference")
    parser.add_argument("--video",      required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--label_map",  default="data/splits/label_map.json")
    parser.add_argument("--config",     default="configs/train_config.yaml")
    parser.add_argument("--base",       default="configs/base_config.yaml")
    parser.add_argument("--top_k",      type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)

    inferencer = SLTInferencer(cfg, args.checkpoint, args.label_map)
    predictions = inferencer.predict(args.video, top_k=args.top_k)

    print(f"\n{'='*45}")
    print(f"  Video: {args.video}")
    print(f"  Top-{args.top_k} Predictions:")
    print(f"{'─'*45}")
    for i, (gloss, conf) in enumerate(predictions, 1):
        bar = "█" * int(conf * 30)
        print(f"  {i}. {gloss:<20s}  {conf:.3f}  {bar}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
