"""
src/models/backbone.py
Visual feature extractor — wraps a CNN backbone to process video frames.

Input:  (B, T, C, H, W)  — batch of video clips
Output: (B, T, feature_dim) — per-frame CNN features
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


BACKBONE_DIMS = {
    "resnet18":        512,
    "resnet50":       2048,
    "resnet101":      2048,
    "efficientnet_b0": 1280,
    "efficientnet_b2": 1408,
}


class VideoBackbone(nn.Module):
    """
    Wraps a torchvision CNN so it processes each frame independently
    and returns per-frame feature vectors.

    The batch and time dimensions are merged for a single CNN forward pass
    (efficient), then split back apart.
    """

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = BACKBONE_DIMS.get(backbone_name, 2048)

        if backbone_name.startswith("resnet"):
            weights = "IMAGENET1K_V1" if pretrained else None
            cnn = getattr(models, backbone_name)(weights=weights)
            # Strip the final classification head — keep only the feature extractor
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])  # output: (B, D, 1, 1)

        elif backbone_name.startswith("efficientnet"):
            weights = "IMAGENET1K_V1" if pretrained else None
            cnn = getattr(models, backbone_name)(weights=weights)
            self.cnn = nn.Sequential(cnn.features, cnn.avgpool)

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. "
                             f"Choose from: {list(BACKBONE_DIMS.keys())}")

        # Freeze backbone initially — unfreeze later during fine-tuning
        self._frozen = False

    def freeze(self) -> None:
        """Freeze backbone weights (use during SSL pretraining)."""
        for p in self.cnn.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze backbone for end-to-end fine-tuning."""
        for p in self.cnn.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)  video frames

        Returns:
            features: (B, T, feature_dim)
        """
        B, T, C, H, W = x.shape

        # Merge batch + time: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # CNN forward: (B*T, feature_dim, 1, 1)
        feats = self.cnn(x)

        # Flatten spatial: (B*T, feature_dim)
        feats = feats.view(B * T, -1)

        # Restore time: (B, T, feature_dim)
        feats = feats.view(B, T, self.feature_dim)

        return feats
