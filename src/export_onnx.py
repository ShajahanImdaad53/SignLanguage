"""
src/export_onnx.py
Export trained model to ONNX format for deployment.

ONNX allows deployment on CPUs, mobile, cloud services without PyTorch.

Usage:
    python src/export_onnx.py \
        --checkpoint models/best_model.pt \
        --config configs/train_config.yaml \
        --output models/slt_model.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

from src.models.slt_model import SLTModel
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("export_onnx")


def export_to_onnx(
    checkpoint_path: str,
    cfg,
    output_path: str = "models/slt_model.onnx",
    opset_version: int = 14,
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to .pt checkpoint
        cfg: Model config
        output_path: Where to save .onnx file
        opset_version: ONNX opset version (14+ for better compatibility)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SLTModel(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"[ONNX] Loaded checkpoint: {checkpoint_path}")

    # Create dummy inputs
    B = 1
    T = cfg.video.max_frames
    H, W = cfg.video.frame_size
    C = cfg.video.channels
    D = cfg.model.keypoint_dim

    dummy_frames = torch.randn(B, T, C, H, W).to(device)
    dummy_keypoints = torch.randn(B, T, D).to(device)

    logger.info(f"[ONNX] Dummy input shapes:")
    logger.info(f"      frames:    {dummy_frames.shape}")
    logger.info(f"      keypoints: {dummy_keypoints.shape}")

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            (dummy_frames, dummy_keypoints),
            str(output_path),
            input_names=["frames", "keypoints"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
        logger.info(f"[ONNX] ✓ Successfully exported to {output_path}")
        logger.info(f"[ONNX] File size: {output_path.stat().st_size / 1e6:.1f} MB")

    except Exception as e:
        logger.error(f"[ONNX] Export failed: {e}")
        raise


def validate_onnx(onnx_path: str) -> bool:
    """Validate ONNX model format."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"[ONNX] ✓ Model validation passed")
        return True
    except ImportError:
        logger.warning("[ONNX] ONNX package not installed, skipping validation")
        return True
    except Exception as e:
        logger.error(f"[ONNX] Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    parser.add_argument("--output", default="models/slt_model.onnx")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)
    export_to_onnx(args.checkpoint, cfg, output_path=args.output, opset_version=args.opset)

    if args.validate:
        validate_onnx(args.output)


if __name__ == "__main__":
    main()
