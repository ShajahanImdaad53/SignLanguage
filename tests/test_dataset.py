"""
tests/test_dataset.py
Unit tests for dataset loading and transforms.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from src.dataset import SLTDataset, build_transforms


@pytest.fixture
def dummy_dataset(tmp_path):
    """Create a minimal dummy dataset for testing."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create dummy frames
    (frames_dir / "video_001").mkdir()
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(frames_dir / "video_001" / f"frame_{i:06d}.jpg")

    # Create dummy CSV
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "video_id,video_path,keypoints_path,label,gloss\n"
        "video_001,dummy.mp4,keypoints.npy,0,hello\n"
    )

    return csv_path, frames_dir


def test_dataset_loading(dummy_dataset, tmp_path):
    """Test that dataset loads and returns correct shapes."""
    csv_path, frames_dir = dummy_dataset

    # Create minimal config
    frames_dir_str = str(frames_dir)  # Capture in outer scope
    class DummyCfg:
        class Video:
            frame_size = [224, 224]
            normalize_mean = [0.485, 0.456, 0.406]
            normalize_std = [0.229, 0.224, 0.225]
            max_frames = 64

        class Model:
            keypoint_dim = 258

        class Data:
            frames_dir = frames_dir_str

        video = Video()
        model = Model()
        data = Data()

    cfg = DummyCfg()

    # Create dataset
    dataset = SLTDataset(str(csv_path), cfg, split="train")
    assert len(dataset) == 1

    # Get one sample
    frames, keypoints, label = dataset[0]

    # Check shapes
    assert frames.shape == (64, 3, 224, 224)  # (T, C, H, W)
    assert keypoints.shape == (64, 258)        # (T, keypoint_dim)
    assert label == 0

    # Check types
    assert isinstance(frames, torch.Tensor)
    assert isinstance(keypoints, torch.Tensor)
    assert isinstance(label, int)


def test_transforms():
    """Test augmentation transforms."""
    cfg_dict = {
        "video": {
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "frame_size": [224, 224],
        }
    }

    class DummyCfg:
        video = type('obj', (object,), cfg_dict["video"])()

    cfg = DummyCfg()

    # Train transforms (with augmentation)
    train_tf = build_transforms(cfg, split="train")
    assert train_tf is not None

    # Val transforms (no augmentation)
    val_tf = build_transforms(cfg, split="val")
    assert val_tf is not None

    # Test on dummy image
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    tensor = train_tf(img)
    assert tensor.shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
