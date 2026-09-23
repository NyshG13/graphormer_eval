"""Hop-masked transformer layer — Experiment 3: Multi-feature Structural Bias.

Builds on Experiment 1 (learnable hop bias) and adds two more structural
signals as additive attention biases:

  sigma_weight * log1p(σ_ij)   — shortest-path count (path redundancy)
  cnbr_weight  * log1p(c_ij)   — common-neighbour count (triangle density)

All three biases are purely structural (topology-derived), orthogonal to the
feature-driven Q·K attention score.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG_INF = float("-inf")

from graphgps.layer.hop_masked_transformer_layer import build_norm
from graphgps.layer.hop_masked_transformer_exp1 import HopBiasMHA


class MultiFeatureBiasMHA(HopBiasMHA):
    """Exp-1 MHA extended with sigma and cnbr biases (Experiment 3).

    Extra learnable scalars:
        self.sigma_weight : scalar weight on log1p(σ_ij)
        self.cnbr_weight  : scalar weight on log1p(|N(i)∩N(j)|)

    Both initialised to 0.0 so they start at the Exp-1 baseline and can
    grow or shrink based on task utility.
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 max_hops: int = 30, dropout: float = 0.0):
        super().__init__(hidden_dim, num_heads, max_hops, dropout)
        # Learnable scalars for the two structural features
        self.sigma_weight = nn.Parameter(torch.zeros(1))
        self.cnbr_weight  = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,             # (B, N, d)
        per_head_mask: torch.Tensor,  # (B, H, N, N) bool
        node_mask: torch.Tensor,      # (B, N) bool
        dist_int: torch.Tensor,       # (B, N, N) long
        sigma: torch.Tensor,          # (B, N, N) float32 — log1p already applied
        cnbr: torch.Tensor,           # (B, N, N) float32 — log1p already applied
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # Hop bias (from Exp 1)
        dist_clamped = dist_int.clamp(0, self.max_hops + 1)
        dist_clamped = torch.where(dist_int < 0,
                                   torch.full_like(dist_int, self.max_hops + 1),
                                   dist_clamped)
        bias = self.hop_bias(dist_clamped)              # (B, N, N, H)
        scores = scores + bias.permute(0, 3, 1, 2)

        # Sigma and common-neighbour biases (broadcast over heads)
        scores = scores + self.sigma_weight * sigma.unsqueeze(1)   # (B,1,N,N)
        scores = scores + self.cnbr_weight  * cnbr.unsqueeze(1)

        # Hop masking
        scores = scores.masked_fill(~per_head_mask, _NEG_INF)

        # Padding masks
        key_pad   = (~node_mask).unsqueeze(1).unsqueeze(2)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(key_pad,   _NEG_INF)
        scores = scores.masked_fill(query_pad, _NEG_INF)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)

        if not self.training:
            self._last_attn         = attn.detach()
            self._last_out_pre_proj = out.detach()

        out = out.transpose(1, 2).reshape(B, N, d)
        return self.out_proj(out)


class MultiFeatureBiasTransformerLayer(nn.Module):
    """Pre-norm encoder layer using MultiFeatureBiasMHA (Experiment 3)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 max_hops: int = 30, dropout: float = 0.1,
                 norm_type: str = "layer"):
        super().__init__()
        self.norm1 = build_norm(norm_type, hidden_dim)
        self.attn  = MultiFeatureBiasMHA(hidden_dim, num_heads, max_hops, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = build_norm(norm_type, hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, per_head_mask, node_mask, dist_int, sigma, cnbr):
        normed   = self.norm1(x, node_mask)
        attn_out = self.attn(normed, per_head_mask, node_mask,
                             dist_int, sigma, cnbr)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x, node_mask)))
        return x
