"""
src/models/transformer.py
Temporal Transformer encoder — models sign sequences over time.

Input:  (B, T, D)  — sequence of per-frame feature vectors
Output: (B, T, hidden_dim) — contextualised sequence representations
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — adds position information
    to frame feature vectors so the transformer knows their order.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Multi-layer Transformer encoder over the time dimension.
    Models long-range dependencies between sign frames.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()

        # Project CNN features to transformer hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,   # (B, T, D) convention
            norm_first=True,    # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)   per-frame features
            src_key_padding_mask:  (B, T) bool mask — True = padded position

        Returns:
            (B, T, hidden_dim)  contextualised sequence
        """
        x = self.input_proj(x)          # (B, T, hidden_dim)
        x = self.pos_encoding(x)        # add position info
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x
