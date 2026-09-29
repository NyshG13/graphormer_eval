# model_hop_masked_transformer_spd.py
"""
Hop-Masked Transformer with Shortest Path Geodesic Embeddings and Continuous Geometric Attention Bias.

Supports 3 Distinct Shortest Path Formulations:
1. Deterministic Shortest Path Node Sequence Encoding (--use_spd_path)
2. Deterministic Shortest Path Edge Sequence Encoding (--use_spd_edge_path)
3. Continuous Geodesic Attention Bias via Floyd-Warshall Metric Embedding (--use_spd_bias)
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATv2Conv

from models import build_node_encoder, build_bond_encoder, LinearBondEncoder

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# 1. Geodesic Metric Attention Bias (Floyd-Warshall Distance Embedding)
# ---------------------------------------------------------------------------
class SPDGeodesicBias(nn.Module):
    """
    Maps discrete shortest-path geodesic distance integers d in {0, 1, ..., max_hops, infinity}
    to head-specific scalar attention biases: Bias_h(u, v) = Embedding_h(dist(u, v)).
    """
    def __init__(self, max_hops: int, num_heads: int):
        super().__init__()
        self.max_hops = max_hops
        self.num_heads = num_heads
        # Embedding for distances 0..max_hops-1, plus index max_hops for disconnected / infinity
        self.spd_bias_emb = nn.Embedding(max_hops + 1, num_heads)
        # Initialize near zero for smooth warm start
        nn.init.normal_(self.spd_bias_emb.weight, mean=0.0, std=0.02)

    def forward(self, dist_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist_masks: (B, K, N, N) float tensor with 1.0 at dist(u, v) == k
        Returns:
            spd_bias: (B, H, N, N) float tensor of head-specific biases
        """
        B, K, N, _ = dist_masks.shape
        H = self.num_heads
        K_eff = min(K, self.max_hops)

        # Reachable pairs in [0..K_eff-1]
        reachable = dist_masks[:, :K_eff].sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # (B, 1, N, N)
        unreachable = 1.0 - reachable                                                # (B, 1, N, N)

        # Vectorized tensor contraction for all reachable distances
        spd_bias = torch.einsum("bknm,kh->bhnm", dist_masks[:, :K_eff], self.spd_bias_emb.weight[:K_eff])

        # Add unreachable / infinity distance bias
        spd_bias = spd_bias + unreachable * self.spd_bias_emb.weight[self.max_hops].view(1, H, 1, 1)
        return spd_bias


# ---------------------------------------------------------------------------
# 2. Deterministic Shortest Path Node Sequence Encoder
# ---------------------------------------------------------------------------
class SPDPathNodeEncoder(nn.Module):
    """Encodes node feature sequences along deterministic shortest paths to enrich base node features."""

    def __init__(
        self,
        hidden_dim: int,
        path_len: int = 3,
        num_paths: int = 20,
        merge_mode: str = "concat",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.path_len = path_len
        self.num_paths = num_paths
        self.seq_len = path_len + 1
        self.merge_mode = merge_mode
        self.scale = hidden_dim ** -0.5

        self.step_embed = nn.Embedding(self.seq_len, hidden_dim)
        self.dist_embed = nn.Embedding(self.seq_len + 10, hidden_dim)

        if merge_mode == "concat":
            self.path_mlp = nn.Sequential(
                nn.Linear(self.seq_len * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif merge_mode == "conv":
            self.conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            )
        else:
            self.path_mlp = nn.Sequential(
                nn.Linear(self.seq_len * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.path_norm = nn.LayerNorm(hidden_dim)
        self.node_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.path_gate = nn.Parameter(torch.tensor(0.0))
        self._diag = None

    def forward(
        self,
        dense_x: torch.Tensor,     # (B, N, d)
        spd_paths: torch.Tensor,   # (B, N, M, seq_len) node indices
        spd_dists: torch.Tensor,   # (B, N, M, seq_len) step distances
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = dense_x.shape
        _, _, M, seq_len = spd_paths.shape

        node_lookup = torch.clamp(spd_paths, min=0, max=N - 1)
        batch_idx = torch.arange(B, device=dense_x.device).view(B, 1, 1, 1).expand(B, N, M, seq_len)
        h_tokens = dense_x[batch_idx, node_lookup]  # (B, N, M, seq_len, d)

        step_ids = torch.arange(seq_len, device=dense_x.device).view(1, 1, 1, seq_len)
        step_emb = self.step_embed(step_ids)
        dist_lookup = torch.clamp(spd_dists, min=0, max=self.dist_embed.num_embeddings - 1)
        dist_emb = self.dist_embed(dist_lookup)

        h_tokens = h_tokens + step_emb + dist_emb

        pad_mask = (spd_paths < 0).unsqueeze(-1)
        h_tokens = h_tokens.masked_fill(pad_mask, 0.0)

        if self.merge_mode == "concat":
            h_flat = h_tokens.view(B, N, M, seq_len * d)
            h_paths = self.path_norm(self.path_mlp(h_flat))
        else:
            h_flat = h_tokens.view(B, N, M, seq_len * d)
            h_paths = self.path_norm(self.path_mlp(h_flat))

        q_node = self.node_q_proj(dense_x).unsqueeze(2)     # (B, N, 1, d)
        k_path = self.path_k_proj(h_paths)                  # (B, N, M, d)
        v_path = self.path_v_proj(h_paths)                  # (B, N, M, d)

        scores = torch.matmul(q_node, k_path.transpose(-2, -1)) * self.scale
        raw_attn = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(raw_attn)

        with torch.no_grad():
            p = raw_attn.squeeze(2).clamp_min(1e-12)
            ent = -(p * p.log()).sum(-1)
            part = 1.0 / p.pow(2).sum(-1).clamp_min(1e-12)
            valid = node_mask if node_mask is not None else torch.ones(B, N, dtype=torch.bool, device=dense_x.device)
            self._diag = {
                "entropy": float(ent[valid].mean().cpu()) if valid.any() else float(ent.mean().cpu()),
                "participation": float(part[valid].mean().cpu()) if valid.any() else float(part.mean().cpu()),
                "top_path": float(p.max(dim=-1)[0][valid].mean().cpu()) if valid.any() else float(p.max(dim=-1)[0].mean().cpu()),
                "gate": float(self.path_gate.detach().cpu()),
            }

        x_path = torch.matmul(attn_weights, v_path).squeeze(2)
        out = dense_x + self.path_gate * self.final_norm(x_path)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# 3. Deterministic Shortest Path Edge Sequence Encoder
# ---------------------------------------------------------------------------
class SPDPathEdgeEncoder(nn.Module):
    """Encodes continuous edge transitions along deterministic shortest paths."""

    def __init__(
        self,
        hidden_dim: int,
        path_len: int = 3,
        num_paths: int = 20,
        merge_mode: str = "concat",
        dropout: float = 0.0,
        dataset_name: str = "PascalVOC-SP",
        edge_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.path_len = path_len
        self.num_paths = num_paths
        self.merge_mode = merge_mode
        self.scale = hidden_dim ** -0.5

        self.bond_encoder = build_bond_encoder(
            hidden_dim, dataset_name=dataset_name, edge_feat_dim=edge_feat_dim
        )
        self.step_embed = nn.Embedding(self.path_len + 1, hidden_dim)

        self.path_mlp = nn.Sequential(
            nn.Linear(self.path_len * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_norm = nn.LayerNorm(hidden_dim)

        self.node_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.path_gate = nn.Parameter(torch.tensor(0.0))
        self._diag = None

    def forward(
        self,
        dense_x: torch.Tensor,
        spd_paths: torch.Tensor,  # (B, N, M, path_len + 1)
        batch_pyg,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = dense_x.shape
        _, _, M, seq_len = spd_paths.shape
        L = seq_len - 1

        if hasattr(batch_pyg, "edge_attr") and batch_pyg.edge_attr is not None:
            edge_emb = self.bond_encoder(batch_pyg.edge_attr)
        else:
            edge_emb = dense_x.new_zeros(batch_pyg.edge_index.size(1), d)

        # Dense edge matrix (B, N, N, d)
        dense_edges = dense_x.new_zeros(B, N, N, d)
        edge_index = batch_pyg.edge_index
        batch_ptr = batch_pyg.ptr if hasattr(batch_pyg, "ptr") else None
        if batch_ptr is not None:
            for b in range(B):
                start_n = int(batch_ptr[b])
                end_n = int(batch_ptr[b + 1])
                e_mask = (edge_index[0] >= start_n) & (edge_index[0] < end_n)
                src = edge_index[0, e_mask] - start_n
                dst = edge_index[1, e_mask] - start_n
                dense_edges[b, src, dst] = edge_emb[e_mask]

        u_idx = torch.clamp(spd_paths[..., :-1], min=0, max=N - 1)
        v_idx = torch.clamp(spd_paths[..., 1:], min=0, max=N - 1)
        batch_idx = torch.arange(B, device=dense_x.device).view(B, 1, 1, 1).expand(B, N, M, L)

        edge_seq = dense_edges[batch_idx, u_idx, v_idx]  # (B, N, M, L, d)
        step_ids = torch.arange(L, device=dense_x.device).view(1, 1, 1, L)
        edge_seq = edge_seq + self.step_embed(step_ids)

        invalid_mask = (spd_paths[..., :-1] < 0) | (spd_paths[..., 1:] < 0)
        edge_seq = edge_seq.masked_fill(invalid_mask.unsqueeze(-1), 0.0)

        h_flat = edge_seq.view(B, N, M, L * d)
        h_paths = self.path_norm(self.path_mlp(h_flat))

        q_node = self.node_q_proj(dense_x).unsqueeze(2)
        k_path = self.path_k_proj(h_paths)
        v_path = self.path_v_proj(h_paths)

        scores = torch.matmul(q_node, k_path.transpose(-2, -1)) * self.scale
        raw_attn = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(raw_attn)

        with torch.no_grad():
            p = raw_attn.squeeze(2).clamp_min(1e-12)
            ent = -(p * p.log()).sum(-1)
            part = 1.0 / p.pow(2).sum(-1).clamp_min(1e-12)
            valid = node_mask if node_mask is not None else torch.ones(B, N, dtype=torch.bool, device=dense_x.device)
            self._diag = {
                "entropy": float(ent[valid].mean().cpu()) if valid.any() else float(ent.mean().cpu()),
                "participation": float(part[valid].mean().cpu()) if valid.any() else float(part.mean().cpu()),
                "top_path": float(p.max(dim=-1)[0][valid].mean().cpu()) if valid.any() else float(p.max(dim=-1)[0].mean().cpu()),
                "gate": float(self.path_gate.detach().cpu()),
            }

        x_edge_path = torch.matmul(attn_weights, v_path).squeeze(2)
        out = dense_x + self.path_gate * self.final_norm(x_edge_path)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# 4. Top-level Transformer Model Supporting SPD Formulations
# ---------------------------------------------------------------------------
from model_hop_masked_transformer_final_3 import (
    HopMaskedTransformerLayer,
    PostGATv2Block,
    build_head_hop_sets,
    _build_alternating_hop_sets,
    compute_gate_aux_loss,
    set_attn_diagnostics,
)


class HopMaskedTransformerModelSPD(nn.Module):
    """Hop-Masked Transformer with Geodesic Metric Attention Bias and Shortest Path Sequence Encodings."""

    def __init__(
        self,
        hidden_dim: int = 120,
        num_heads: int = 12,
        ffn_ratio: float = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_hops: int = 12,
        hop_mode: str = "single",
        hop_window: int = 1,
        hop_file: Optional[str] = None,
        num_global_heads: int = 1,
        output_dim: int = 21,
        graph_pool: str = "sum",
        task_level: str = "node",
        dataset_name: str = "PascalVOC-SP",
        node_feat_dim: Optional[int] = None,
        lap_pe_dim: int = 0,
        block_diag_out: bool = True,
        dynamic_cross_hop: bool = True,
        norm_type: str = "graph",
        v_head_dim: Optional[int] = None,
        mask_type: str = "shortest_path",
        adj_self_loops: bool = False,
        use_moe_gating: bool = False,
        top_k: int = 0,
        gate_noise: float = 0.1,
        balance_coeff: float = 0.01,
        entropy_coeff: float = 0.01,
        use_virtual_node: bool = False,
        num_post_gat_layers: int = 0,
        num_gat_heads: int = 4,
        cross_hop_hop_embedding: bool = False,
        use_edge_features: bool = False,
        edge_feat_dim: Optional[int] = None,
        cross_hop_no_ffn: bool = False,
        blend_adj_power: bool = False,
        use_edge_bias: bool = False,
        use_rrwp: bool = False,
        rrwp_dim: int = 8,
        # SPD specific flags
        use_spd_bias: bool = False,
        use_spd_path: bool = False,
        use_spd_edge_path: bool = False,
        spd_num_paths: int = 20,
        spd_path_len: int = 3,
        spd_merge: str = "concat",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_hops = max_hops
        self.task_level = task_level
        self.graph_pool = graph_pool
        self.mask_type = mask_type
        self.adj_self_loops = adj_self_loops
        self.use_moe_gating = use_moe_gating
        self.use_virtual_node = use_virtual_node
        self.use_alternating = (hop_mode == "alternating")
        self.use_spd_bias = use_spd_bias
        self.use_spd_path = use_spd_path
        self.use_spd_edge_path = use_spd_edge_path

        self.node_encoder = build_node_encoder(
            hidden_dim,
            dataset_name=dataset_name,
            lap_pe_dim=lap_pe_dim,
            node_feat_dim=node_feat_dim,
        )

        # 1. Geodesic Metric Attention Bias Module
        self.spd_bias_module = None
        if use_spd_bias:
            self.spd_bias_module = SPDGeodesicBias(max_hops=max_hops, num_heads=num_heads)

        # 2. Shortest Path Node Sequence Encoder
        self.spd_node_encoder = None
        if use_spd_path:
            self.spd_node_encoder = SPDPathNodeEncoder(
                hidden_dim=hidden_dim,
                path_len=spd_path_len,
                num_paths=spd_num_paths,
                merge_mode=spd_merge,
                dropout=dropout,
            )

        # 3. Shortest Path Edge Sequence Encoder
        self.spd_edge_encoder = None
        if use_spd_edge_path:
            self.spd_edge_encoder = SPDPathEdgeEncoder(
                hidden_dim=hidden_dim,
                path_len=spd_path_len,
                num_paths=spd_num_paths,
                merge_mode=spd_merge,
                dropout=dropout,
                dataset_name=dataset_name,
                edge_feat_dim=edge_feat_dim,
            )

        if self.use_alternating:
            self.per_layer_hop_sets = _build_alternating_hop_sets(
                max_hops=max_hops, num_heads=num_heads,
                num_layers=num_layers, num_global_heads=num_global_heads,
            )
            self.head_hop_sets = self.per_layer_hop_sets[0]
        else:
            self.head_hop_sets = build_head_hop_sets(
                max_hops=max_hops, num_heads=num_heads,
                mode=hop_mode, window=hop_window,
                hop_file=hop_file, num_global_heads=num_global_heads,
            )

        ffn_dim = int(hidden_dim * ffn_ratio)
        self.layers = nn.ModuleList([
            HopMaskedTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                block_diag_out=block_diag_out,
                dynamic_cross_hop=dynamic_cross_hop,
                norm_type=norm_type,
                v_head_dim=v_head_dim,
                use_moe_gating=use_moe_gating,
                max_hops=max_hops,
                top_k=top_k,
                gate_noise=gate_noise,
                cross_hop_hop_embedding=cross_hop_hop_embedding,
                cross_hop_no_ffn=cross_hop_no_ffn,
                blend_adj_power=blend_adj_power,
                use_edge_bias=use_edge_bias or use_spd_bias,
                use_rrwp=use_rrwp,
            )
            for _ in range(num_layers)
        ])

        self.post_gat = None
        if num_post_gat_layers > 0:
            self.post_gat = PostGATv2Block(
                hidden_dim=hidden_dim,
                num_heads=num_gat_heads,
                num_layers=num_post_gat_layers,
                dropout=dropout,
            )

        if task_level == "graph":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head = nn.Linear(hidden_dim, output_dim)

    def _build_per_head_mask(
        self,
        dist_masks: torch.Tensor,
        hop_sets: Optional[List[Optional[List[int]]]] = None,
    ) -> torch.Tensor:
        B, K_runtime, N, _ = dist_masks.shape
        H = self.num_heads
        if hop_sets is None:
            hop_sets = self.head_hop_sets
        out = dist_masks.new_zeros(B, H, N, N, dtype=torch.bool)
        for h, hop_set in enumerate(hop_sets):
            if hop_set is None:
                out[:, h] = True
                continue
            idx = [k for k in hop_set if k < K_runtime]
            if not idx:
                out[:, h] = True
                continue
            stacked = dist_masks[:, idx].bool().any(dim=1)
            out[:, h] = stacked

        eye = torch.eye(N, device=dist_masks.device, dtype=torch.bool)
        non_self = out & ~eye[None, None]
        has_nonself = non_self.any(dim=-1).any(dim=-1)
        out[~has_nonself] = True
        return out

    def get_path_diagnostics(self):
        if self.spd_edge_encoder is not None:
            return self.spd_edge_encoder._diag
        if self.spd_node_encoder is not None:
            return self.spd_node_encoder._diag
        return None

    def forward(
        self,
        batch,
        dist_masks: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        spd_paths: Optional[torch.Tensor] = None,
        spd_dists: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lap_pe = getattr(batch, "lap_pe", None)
        edge_attr = getattr(batch, "edge_attr", None)
        edge_index = getattr(batch, "edge_index", None)
        h = self.node_encoder(batch.x, edge_index, edge_attr, lap_pe=lap_pe)
        dense_x, mask = to_dense_batch(h, batch.batch)
        if node_mask is None:
            node_mask = mask

        # 1. Deterministic Shortest Path Node Sequence Enrichment
        if self.use_spd_path and spd_paths is not None and self.spd_node_encoder is not None:
            dense_x = self.spd_node_encoder(dense_x, spd_paths, spd_dists, node_mask=node_mask)

        # 2. Deterministic Shortest Path Edge Sequence Enrichment
        if self.use_spd_edge_path and spd_paths is not None and self.spd_edge_encoder is not None:
            dense_x = self.spd_edge_encoder(dense_x, spd_paths, batch, node_mask=node_mask)

        # 3. Continuous Geodesic Metric Attention Bias
        spd_bias = None
        if self.use_spd_bias and self.spd_bias_module is not None:
            spd_bias = self.spd_bias_module(dist_masks)

        # Pass through Transformer Stack with per-head masks
        for i, layer in enumerate(self.layers):
            hop_sets = self.per_layer_hop_sets[i] if self.use_alternating else None
            per_head_mask = self._build_per_head_mask(dist_masks, hop_sets=hop_sets)
            dense_x = layer(
                dense_x,
                per_head_mask,
                node_mask=node_mask,
                edge_bias=spd_bias,
            )

        if self.task_level == "node":
            out_flat = dense_x[mask]
            if self.post_gat is not None:
                out_flat = self.post_gat(out_flat, batch.edge_index)
            return self.head(out_flat)

        mask_f = mask.unsqueeze(-1).float()
        if self.graph_pool == "mean":
            g_emb = (dense_x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        else:
            g_emb = (dense_x * mask_f).sum(dim=1)
        return self.head(g_emb)
