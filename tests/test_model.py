"""
tests/test_model.py
Unit tests for model architecture and forward pass.
"""

import pytest
import torch

from src.models.slt_model import SLTModel, build_model


@pytest.fixture
def dummy_config():
    """Create a minimal config for testing."""
    class DummyCfg:
        class Model:
            name = "hybrid_ssl_transformer"
            backbone = "resnet50"
            pretrained = False
            hidden_dim = 128
            num_heads = 4
            num_layers = 2
            dropout = 0.1
            vocab_size = 500
            max_seq_len = 64
            feature_dim = 2048
            keypoint_dim = 258
            fusion = "concat"

        class Training:
            batch_size = 4
            learning_rate = 0.001
            gradient_clip = 1.0

        class SSL:
            method = "masked_autoencoder"

        model = Model()
        training = Training()
        ssl = SSL()

    return DummyCfg()


def test_model_creation(dummy_config):
    """Test that model initializes correctly."""
    model = SLTModel(dummy_config)
    assert model is not None
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")


def test_forward_finetune(dummy_config):
    """Test finetune forward pass."""
    device = torch.device("cpu")
    model = SLTModel(dummy_config).to(device)
    model.eval()

    B, T = 2, 16
    frames = torch.randn(B, T, 3, 224, 224).to(device)
    keypoints = torch.randn(B, T, dummy_config.model.keypoint_dim).to(device)

    with torch.no_grad():
        output = model(frames, keypoints, mode="finetune")

    assert "logits" in output
    assert output["logits"].shape == (B, dummy_config.model.vocab_size)


def test_forward_pretrain(dummy_config):
    """Test SSL pretraining forward pass."""
    device = torch.device("cpu")
    model = SLTModel(dummy_config).to(device)
    model.eval()

    B, T = 2, 16
    frames = torch.randn(B, T, 3, 224, 224).to(device)
    keypoints = torch.randn(B, T, dummy_config.model.keypoint_dim).to(device)
    mask = torch.ones(B, T, dtype=torch.bool).to(device)
    mask[:, :T//2] = False  # Mask first half

    with torch.no_grad():
        output = model(frames, keypoints, mask=mask, mode="pretrain")

    assert "reconstructed" in output
    assert output["reconstructed"].shape == (B, T, dummy_config.model.feature_dim)


def test_backbone_freeze_unfreeze(dummy_config):
    """Test backbone freeze/unfreeze mechanism."""
    model = SLTModel(dummy_config)

    # Initially trainable
    assert any(p.requires_grad for p in model.backbone.parameters())

    # Freeze
    model.backbone.freeze()
    assert not any(p.requires_grad for p in model.backbone.parameters())

    # Unfreeze
    model.backbone.unfreeze()
    assert any(p.requires_grad for p in model.backbone.parameters())


def test_build_model(dummy_config):
    """Test the build_model factory."""
    model = build_model(dummy_config)
    assert model is not None
    assert isinstance(model, SLTModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
