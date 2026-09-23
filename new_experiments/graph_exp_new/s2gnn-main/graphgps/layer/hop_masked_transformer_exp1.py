"""Hop-masked transformer layer — Experiment 1: Learnable Hop Bias.

Same architecture as the baseline HopMaskedTransformerLayer, but with a
per-hop learnable scalar bias added to the attention scores before softmax.

Key addition in HopMaskedMHA:
    self.hop_bias = nn.Embedding(max_hops + 1, num_heads)

During forward, the integer distance matrix (dist_int, shape B×N×N, values
0..max_hops, -1 mapped to 0) is used to look up a scalar bias per head.
This bias is PURELY STRUCTURAL — it depends only on hop distance, not on
node features — so it carries a different signal than the Q·K dot product.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Reuse helpers from the baseline layer (no duplication).
# ---------------------------------------------------------------------------
from graphgps.layer.hop_masked_transformer_layer import (
    _split_contiguous,
    build_head_hop_sets,
    LayerNormWrap,
    build_norm,
)


# ---------------------------------------------------------------------------
# Exp-1 MHA: adds learnable hop-distance embedding as attention bias.
# ---------------------------------------------------------------------------
class HopBiasMHA(nn.Module):
    """Multi-head attention with per-head hop masking AND a learnable hop bias.

    The bias is a scalar per (hop_distance, head) pair — an Embedding table of
    shape (max_hops+2, num_heads).  Index 0 = self-loop, index k = hop k,
    index max_hops+1 = padding/unreachable (clamped to 0 bias).

    This bias adds purely structural information on top of the feature-based
    Q·K score, answering 'is hop distance k intrinsically important for this
    task?' — orthogonal to what attention already learns.
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 max_hops: int = 30, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.max_hops   = max_hops

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Learnable hop bias: (max_hops+2) embeddings, one per head.
        # Extra slot (max_hops+1) is for padding/unreachable — initialised
        # to 0 and its gradient will naturally stay near 0 since those
        # positions are masked to -inf regardless.
        self.hop_bias = nn.Embedding(max_hops + 2, num_heads,
                                     padding_idx=max_hops + 1)
        nn.init.zeros_(self.hop_bias.weight)

    def forward(
        self,
        x: torch.Tensor,             # (B, N, d)
        per_head_mask: torch.Tensor,  # (B, H, N, N) bool
        node_mask: torch.Tensor,      # (B, N) bool
        dist_int: torch.Tensor,       # (B, N, N) long, -1 = unreachable/pad
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # ------------------------------------------------------------------
        # Learnable hop bias — clamp unreachable (-1) to padding index.
        # ------------------------------------------------------------------
        dist_clamped = dist_int.clamp(0, self.max_hops + 1)
        # Mark unreachable as padding idx so their bias = 0
        dist_clamped = torch.where(dist_int < 0,
                                   torch.full_like(dist_int, self.max_hops + 1),
                                   dist_clamped)
        bias = self.hop_bias(dist_clamped)       # (B, N, N, H)
        scores = scores + bias.permute(0, 3, 1, 2)  # (B, H, N, N)

        # Hop masking (hard — outside-hop positions become -inf)
        scores = scores.masked_fill(~per_head_mask, _NEG_INF)

        # Padding masks
        key_pad   = (~node_mask).unsqueeze(1).unsqueeze(2)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(key_pad,   _NEG_INF)
        scores = scores.masked_fill(query_pad, _NEG_INF)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                 # (B, H, N, Dh)

        if not self.training:
            self._last_attn         = attn.detach()
            self._last_out_pre_proj = out.detach()

        out = out.transpose(1, 2).reshape(B, N, d)
        return self.out_proj(out)


class HopBiasTransformerLayer(nn.Module):
    """Pre-norm encoder layer using HopBiasMHA (Experiment 1)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 max_hops: int = 30, dropout: float = 0.1,
                 norm_type: str = "layer"):
        super().__init__()
        self.norm1 = build_norm(norm_type, hidden_dim)
        self.attn  = HopBiasMHA(hidden_dim, num_heads, max_hops, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = build_norm(norm_type, hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, per_head_mask, node_mask, dist_int):
        normed   = self.norm1(x, node_mask)
        attn_out = self.attn(normed, per_head_mask, node_mask, dist_int)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x, node_mask)))
        return x
