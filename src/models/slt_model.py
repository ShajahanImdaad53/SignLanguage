"""
src/models/slt_model.py
Full Hybrid SSL Sign Language Translation Model.

Architecture:
    VideoBackbone (CNN per-frame) ──┐
                                    ├─► FusionLayer ─► TemporalTransformer ─► Classifier
    KeypointEncoder (MLP) ──────────┘

The model supports two modes:
    - 'pretrain': SSL pretraining (masked autoencoder / contrastive)
    - 'finetune': Supervised sign classification / translation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .backbone import VideoBackbone
from .transformer import TemporalTransformer


class KeypointEncoder(nn.Module):
    """
    Small MLP to encode MediaPipe keypoints (hand + pose landmarks)
    into a fixed-size vector matching hidden_dim.
    """

    def __init__(self, keypoint_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(keypoint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, keypoint_dim) → (B, T, hidden_dim)"""
        return self.net(x)


class FusionLayer(nn.Module):
    """
    Fuse visual (CNN) features and keypoint features.
    Supports: concat → project | attention | gated
    """

    def __init__(self, visual_dim: int, hidden_dim: int, fusion: str = "concat"):
        super().__init__()
        self.fusion = fusion

        if fusion == "concat":
            # Both streams go to hidden_dim before concat
            self.visual_proj = nn.Linear(visual_dim, hidden_dim)
            self.fusion_proj  = nn.Linear(hidden_dim * 2, hidden_dim)

        elif fusion == "gated":
            self.visual_proj  = nn.Linear(visual_dim, hidden_dim)
            self.gate         = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        elif fusion == "attention":
            self.visual_proj  = nn.Linear(visual_dim, hidden_dim)
            self.attn         = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

    def forward(self, visual: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual:    (B, T, visual_dim)  — CNN features
            keypoints: (B, T, hidden_dim)  — encoded keypoints

        Returns:
            (B, T, hidden_dim)  — fused features
        """
        v = self.visual_proj(visual)  # (B, T, hidden_dim)

        if self.fusion == "concat":
            x = torch.cat([v, keypoints], dim=-1)  # (B, T, 2*hidden_dim)
            return self.fusion_proj(x)

        elif self.fusion == "gated":
            combined = torch.cat([v, keypoints], dim=-1)
            gate = self.gate(combined)
            return gate * v + (1 - gate) * keypoints

        elif self.fusion == "attention":
            out, _ = self.attn(query=v, key=keypoints, value=keypoints)
            return out + v   # residual


class SLTModel(nn.Module):
    """
    Hybrid SSL Sign Language Translation Model.

    Modes:
        mode='finetune': returns (logits,)  shape (B, vocab_size)
        mode='pretrain': returns reconstruction targets for SSL loss
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m = cfg.model

        # 1. Visual backbone
        self.backbone = VideoBackbone(
            backbone_name=m.backbone,
            pretrained=m.pretrained,
        )
        visual_dim = self.backbone.feature_dim  # e.g. 2048 for resnet50

        # 2. Keypoint encoder
        self.keypoint_encoder = KeypointEncoder(
            keypoint_dim=m.keypoint_dim,
            hidden_dim=m.hidden_dim,
            dropout=m.dropout,
        )

        # 3. Fusion
        self.fusion = FusionLayer(
            visual_dim=visual_dim,
            hidden_dim=m.hidden_dim,
            fusion=m.get("fusion", "concat"),
        )

        # 4. Temporal transformer
        self.temporal_encoder = TemporalTransformer(
            input_dim=m.hidden_dim,
            hidden_dim=m.hidden_dim,
            num_heads=m.num_heads,
            num_layers=m.num_layers,
            dropout=m.dropout,
            max_seq_len=m.max_seq_len,
        )

        # 5. Classification head (fine-tuning)
        self.classifier = nn.Sequential(
            nn.LayerNorm(m.hidden_dim),
            nn.Linear(m.hidden_dim, m.vocab_size),
        )

        # 6. SSL decoder head (pretraining — reconstruct masked frames)
        self.ssl_decoder = nn.Sequential(
            nn.Linear(m.hidden_dim, m.hidden_dim),
            nn.GELU(),
            nn.Linear(m.hidden_dim, visual_dim),   # reconstruct CNN features
        )

        self.hidden_dim = m.hidden_dim

    def encode(
        self,
        frames: torch.Tensor,
        keypoints: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shared encoder used by both pretrain and finetune.

        Args:
            frames:    (B, T, C, H, W)
            keypoints: (B, T, keypoint_dim)
            mask:      (B, T) bool — True = masked position (SSL only)

        Returns:
            (B, T, hidden_dim)
        """
        # Visual features
        visual_feats = self.backbone(frames)        # (B, T, visual_dim)

        # Keypoint features
        kp_feats = self.keypoint_encoder(keypoints) # (B, T, hidden_dim)

        # Fusion
        fused = self.fusion(visual_feats, kp_feats) # (B, T, hidden_dim)

        # Apply mask tokens for SSL (replace masked positions with learned token)
        if mask is not None:
            # mask: True = masked. We zero out masked positions here;
            # the decoder tries to reconstruct them.
            fused = fused * (~mask).unsqueeze(-1).float()

        # Temporal modelling
        encoded = self.temporal_encoder(fused)      # (B, T, hidden_dim)

        return encoded

    def forward(
        self,
        frames: torch.Tensor,
        keypoints: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = "finetune",
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames:    (B, T, C, H, W)
            keypoints: (B, T, keypoint_dim)
            mask:      (B, T) bool — masked positions for SSL
            mode:      'finetune' | 'pretrain'

        Returns:
            dict with 'logits' (finetune) or 'reconstructed' (pretrain)
        """
        encoded = self.encode(frames, keypoints, mask=mask)

        if mode == "finetune":
            # Mean-pool over time, classify
            pooled = encoded.mean(dim=1)           # (B, hidden_dim)
            logits = self.classifier(pooled)       # (B, vocab_size)
            return {"logits": logits}

        elif mode == "pretrain":
            # Decode all positions — loss computed only on masked ones
            reconstructed = self.ssl_decoder(encoded)  # (B, T, visual_dim)
            return {"reconstructed": reconstructed}

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'finetune' or 'pretrain'")


def build_model(cfg) -> SLTModel:
    """Convenience factory used by trainer scripts."""
    model = SLTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,} | Trainable: {trainable:,}")
    return model
