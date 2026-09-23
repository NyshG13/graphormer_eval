"""Hop-masked transformer layer — Experiment 4: Path-Aware Attention.

Core idea (from TA): the point of PathNet is NOT to use path statistics but
to propagate information THROUGH intermediate nodes.  This layer implements a
lightweight version where the attention VALUE for each pair (i, j) is
augmented by the features of the intermediate node m on the shortest path
i → m → ... → j:

    V_effective[i, j] = V[j] + β · W_path(x[m])

where m = path_mid[i, j] (precomputed BFS parent pointer, stored in batch).

Architecture
------------
- A configurable number of the restricted heads (``num_path_heads``) use this
  augmented value mechanism.  The remaining heads are unchanged.
- One extra linear W_path ∈ ℝ^{d × Dh} per path head (here shared across
  all path heads for parameter efficiency).
- One learnable scalar β (path_beta), initialised to 0 so the model starts
  at the baseline and grows path influence only if it helps.

Parameter overhead
------------------
W_path : hidden_dim × head_dim  =  240 × 8  =  1 920  params  (for 30 heads)
path_beta : 1  scalar
Total: ~1921 extra parameters — < 0.1% of a typical model.

Memory note
-----------
The gather of x[path_mid] is done on the dense (B, N_max, N_max) index tensor.
For Peptides-func (N_max ≈ 444, B=200): gather output is (B, N, N, Dh) =
200 × 444 × 444 × 8 floats ≈ 630 MB.  If this is too large for a given GPU,
reduce batch size or set ``num_path_heads=1``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG_INF = float("-inf")

from graphgps.layer.hop_masked_transformer_layer import build_norm
from graphgps.layer.hop_masked_transformer_exp1 import HopBiasMHA


class PathAwareMHA(nn.Module):
    """Multi-head attention where selected heads use path-augmented values.

    Parameters
    ----------
    hidden_dim : int
    num_heads : int
    max_hops : int
    num_path_heads : int
        How many of the restricted heads (starting from head 0) use the
        path-augmented value.  Set to 0 to fall back to Exp-1 behaviour.
    dropout : float
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 max_hops: int = 30, num_path_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim    = hidden_dim
        self.num_heads     = num_heads
        self.head_dim      = hidden_dim // num_heads
        self.max_hops      = max_hops
        self.num_path_heads = num_path_heads

        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Learnable hop bias (from Exp 1) — keep it so Exp 4 is strictly
        # more expressive than the baseline.
        self.hop_bias = nn.Embedding(max_hops + 2, num_heads,
                                     padding_idx=max_hops + 1)
        nn.init.zeros_(self.hop_bias.weight)

        # Path augmentation
        if num_path_heads > 0:
            # Shared projection for intermediate-node features → head_dim
            self.path_proj = nn.Linear(hidden_dim, self.head_dim, bias=False)
            # Learnable mixing scalar; init=0 so we start at baseline
            self.path_beta = nn.Parameter(torch.zeros(1))
        else:
            self.path_proj = None
            self.path_beta = None

    # ------------------------------------------------------------------
    # Helper: gather intermediate-node features for all (i,j) pairs
    # ------------------------------------------------------------------
    def _gather_intermediate(
        self,
        x_dense: torch.Tensor,   # (B, N, d)  — dense node features
        path_mid: torch.Tensor,  # (B, N, N) long — global node index or -1
    ) -> torch.Tensor:
        """Return projected intermediate-node features, shape (B, N, N, Dh).

        For pairs where path_mid = -1 (no intermediate / 1-hop / pad), the
        returned value is the zero vector.
        """
        B, N, d = x_dense.shape
        Dh = self.head_dim

        # Clamp -1 → 0 for safe indexing; we'll zero those out afterwards.
        valid = path_mid >= 0                                    # (B, N, N) bool
        idx   = path_mid.clamp(min=0)                           # (B, N, N) long

        # Gather x at intermediate node positions
        # x_dense: (B, N, d) → need (B, N, N, d)
        # Flatten batch×pair dim to do a single gather
        B_N = B * N * N
        idx_flat = idx.reshape(B_N)                             # (B*N*N,)
        # We need to pick from the correct graph in the batch.
        # batch_offset[b] = b*N shifts idx into the (B*N, d) view.
        # Since path_mid already stores GLOBAL (batched) indices, we index
        # into the flattened (B*N, d) tensor directly.
        x_flat = x_dense.reshape(B * N, d)                     # (B*N, d)

        # For path_mid values that are already global node indices in the
        # batch (as set by pad_path_features), we can index x_flat directly:
        mid_feats = x_flat[idx_flat]                            # (B*N*N, d)
        mid_feats = mid_feats.reshape(B, N, N, d)

        # Project to head_dim
        mid_proj = self.path_proj(mid_feats)                    # (B, N, N, Dh)

        # Zero out invalid positions
        mid_proj = mid_proj * valid.unsqueeze(-1).float()
        return mid_proj

    def forward(
        self,
        x: torch.Tensor,             # (B, N, d)
        per_head_mask: torch.Tensor,  # (B, H, N, N) bool
        node_mask: torch.Tensor,      # (B, N) bool
        dist_int: torch.Tensor,       # (B, N, N) long, -1 = unreachable
        path_mid: torch.Tensor,       # (B, N, N) long, -1 = no intermediate
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)   # (B,H,N,Dh)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)   # (B,H,N,Dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # Hop bias
        dist_clamped = dist_int.clamp(0, self.max_hops + 1)
        dist_clamped = torch.where(dist_int < 0,
                                   torch.full_like(dist_int, self.max_hops + 1),
                                   dist_clamped)
        bias = self.hop_bias(dist_clamped)                     # (B, N, N, H)
        scores = scores + bias.permute(0, 3, 1, 2)

        # Hop masking
        scores = scores.masked_fill(~per_head_mask, _NEG_INF)
        key_pad   = (~node_mask).unsqueeze(1).unsqueeze(2)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(key_pad,   _NEG_INF)
        scores = scores.masked_fill(query_pad, _NEG_INF)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)                             # (B, H, N, N)

        # ------------------------------------------------------------------
        # Standard output for all heads
        # out[b,h,i,:] = Σ_j attn[b,h,i,j] * v[b,h,j,:]
        # ------------------------------------------------------------------
        out = torch.matmul(attn, v)                             # (B, H, N, Dh)

        # ------------------------------------------------------------------
        # Path augmentation for the first num_path_heads restricted heads.
        # For head h in [0, num_path_heads):
        #   out_aug[b,h,i,:] += β * Σ_j attn[b,h,i,j] * path_proj(x[m_ij])
        # where m_ij = path_mid[b,i,j].
        # ------------------------------------------------------------------
        if self.num_path_heads > 0 and self.path_proj is not None:
            # (B, N, N, Dh) — projected intermediate-node features
            mid_proj = self._gather_intermediate(x, path_mid)

            # For each path head h, compute:
            #   Σ_j attn[b,h,i,j] * mid_proj[b,i,j,:]
            # = einsum('bhij, bijd -> bhid', attn_path, mid_proj)
            attn_path = attn[:, :self.num_path_heads]           # (B, Hp, N, N)
            # (B, Hp, N, N) × (B, N, N, Dh) → (B, Hp, N, Dh)
            path_contribution = torch.einsum(
                'bhij, bijd -> bhid', attn_path, mid_proj
            )
            out[:, :self.num_path_heads] = (
                out[:, :self.num_path_heads]
                + self.path_beta * path_contribution
            )

        if not self.training:
            self._last_attn         = attn.detach()
            self._last_out_pre_proj = out.detach()

        out = out.transpose(1, 2).reshape(B, N, d)
        return self.out_proj(out)


class PathAwareTransformerLayer(nn.Module):
    """Pre-norm encoder layer using PathAwareMHA (Experiment 4)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 max_hops: int = 30, num_path_heads: int = 4,
                 dropout: float = 0.1, norm_type: str = "layer"):
        super().__init__()
        self.norm1 = build_norm(norm_type, hidden_dim)
        self.attn  = PathAwareMHA(hidden_dim, num_heads, max_hops,
                                  num_path_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = build_norm(norm_type, hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, per_head_mask, node_mask, dist_int, path_mid):
        normed   = self.norm1(x, node_mask)
        attn_out = self.attn(normed, per_head_mask, node_mask,
                             dist_int, path_mid)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x, node_mask)))
        return x
