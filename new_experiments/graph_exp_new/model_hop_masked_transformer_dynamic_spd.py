# model_hop_masked_transformer_dynamic_spd.py
"""
Hop-Masked Transformer with Dynamic Max-Diameter Padded Shortest Path Sequence Encodings.

For every batch, the actual maximum shortest-path geodesic distance is dynamically measured,
and all sampled shortest paths in the batch are padded up to this dynamic batch maximum.
Masked sequence pooling and path attention ensure scale-invariant representations across
arbitrary path lengths without fixed horizon truncation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from models import build_node_encoder, build_bond_encoder

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Dynamic Max-Diameter Padded Shortest Path Node Sequence Encoder
# ---------------------------------------------------------------------------
class SPDPathDynamicNodeEncoder(nn.Module):
    """Encodes variable-length shortest path sequences padded to dynamic batch diameter."""

    def __init__(
        self,
        hidden_dim: int,
        num_paths: int = 8,
        max_diameter_cap: int = 40,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_paths = num_paths
        self.scale = hidden_dim ** -0.5

        # Learnable step and distance positional encodings
        self.step_embed = nn.Embedding(max_diameter_cap + 5, hidden_dim)
        self.dist_embed = nn.Embedding(max_diameter_cap + 5, hidden_dim)

        # Path sequence refinement MLP (applied to masked pooled path tokens)
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_norm = nn.LayerNorm(hidden_dim)

        # Multi-path attention aggregator (aggregates K shortest paths per node)
        self.node_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Learnable residual gate initialized to 0.0
        self.path_gate = nn.Parameter(torch.tensor(0.0))
        self._diag = None

    def forward(
        self,
        dense_x: torch.Tensor,     # (B, N, d)
        spd_paths: torch.Tensor,   # (B, N, K, L_batch) node indices along shortest paths (-1 for pad)
        spd_dists: torch.Tensor,   # (B, N, K, L_batch) step distances (-1 for pad)
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = dense_x.shape
        _, _, K, L_batch = spd_paths.shape

        # Lookup node embeddings along the path
        node_lookup = torch.clamp(spd_paths, min=0, max=N - 1)
        batch_idx = torch.arange(B, device=dense_x.device).view(B, 1, 1, 1).expand(B, N, K, L_batch)
        h_tokens = dense_x[batch_idx, node_lookup]  # (B, N, K, L_batch, d)

        # Positional and distance embeddings
        step_ids = torch.arange(L_batch, device=dense_x.device).view(1, 1, 1, L_batch)
        step_ids = torch.clamp(step_ids, min=0, max=self.step_embed.num_embeddings - 1)
        step_emb = self.step_embed(step_ids)

        dist_lookup = torch.clamp(spd_dists, min=0, max=self.dist_embed.num_embeddings - 1)
        dist_emb = self.dist_embed(dist_lookup)

        h_tokens = self.token_proj(h_tokens + step_emb + dist_emb)

        # Mask valid (non-padded) tokens
        valid_token_mask = (spd_paths >= 0).unsqueeze(-1).float()  # (B, N, K, L_batch, 1)
        h_tokens = h_tokens * valid_token_mask

        # Dynamic Masked Mean-Pooling along the variable-length path dimension (L_batch -> 1)
        token_counts = valid_token_mask.sum(dim=-2).clamp_min(1.0)  # (B, N, K, 1)
        h_paths = self.path_norm(h_tokens.sum(dim=-2) / token_counts)  # (B, N, K, d)

        # Path-to-Node Attention Aggregation
        q_node = self.node_q_proj(dense_x).unsqueeze(2)     # (B, N, 1, d)
        k_path = self.path_k_proj(h_paths)                  # (B, N, K, d)
        v_path = self.path_v_proj(h_paths)                  # (B, N, K, d)

        scores = torch.matmul(q_node, k_path.transpose(-2, -1)) * self.scale  # (B, N, 1, K)
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
                "dynamic_L": int(L_batch),
            }

        x_path = torch.matmul(attn_weights, v_path).squeeze(2)  # (B, N, d)
        out = dense_x + self.path_gate * self.final_norm(x_path)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Dynamic Max-Diameter Padded Shortest Path Edge Sequence Encoder
# ---------------------------------------------------------------------------
class SPDPathDynamicEdgeEncoder(nn.Module):
    """Encodes edge feature sequences along variable-length shortest paths padded to dynamic batch diameter."""

    def __init__(
        self,
        hidden_dim: int,
        num_paths: int = 8,
        max_diameter_cap: int = 40,
        dropout: float = 0.0,
        dataset_name: str = "PascalVOC-SP",
        edge_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_paths = num_paths
        self.scale = hidden_dim ** -0.5

        self.bond_encoder = build_bond_encoder(
            hidden_dim, dataset_name=dataset_name, edge_feat_dim=edge_feat_dim
        )
        self.step_embed = nn.Embedding(max_diameter_cap + 5, hidden_dim)

        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        spd_paths: torch.Tensor,  # (B, N, K, L_batch)
        batch_pyg,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = dense_x.shape
        _, _, K, L_batch = spd_paths.shape

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

        # Edges between consecutive path nodes (L_trans = L_batch - 1)
        L_trans = max(L_batch - 1, 1)
        u_idx = torch.clamp(spd_paths[..., :-1], min=0, max=N - 1)
        v_idx = torch.clamp(spd_paths[..., 1:], min=0, max=N - 1)
        batch_idx = torch.arange(B, device=dense_x.device).view(B, 1, 1, 1).expand(B, N, K, L_trans)

        edge_seq = dense_edges[batch_idx, u_idx, v_idx]  # (B, N, K, L_trans, d)
        step_ids = torch.arange(L_trans, device=dense_x.device).view(1, 1, 1, L_trans)
        step_ids = torch.clamp(step_ids, min=0, max=self.step_embed.num_embeddings - 1)
        edge_seq = self.token_proj(edge_seq + self.step_embed(step_ids))

        # Mask valid edge transitions
        valid_trans_mask = ((spd_paths[..., :-1] >= 0) & (spd_paths[..., 1:] >= 0)).unsqueeze(-1).float()
        edge_seq = edge_seq * valid_trans_mask

        # Dynamic Masked Mean Pooling over transitions
        trans_counts = valid_trans_mask.sum(dim=-2).clamp_min(1.0)
        h_paths = self.path_norm(edge_seq.sum(dim=-2) / trans_counts)  # (B, N, K, d)

        # Path-to-Node Attention Aggregation
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
                "dynamic_L": int(L_batch),
            }

        x_edge_path = torch.matmul(attn_weights, v_path).squeeze(2)
        out = dense_x + self.path_gate * self.final_norm(x_edge_path)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Top-level Hop-Masked Transformer with Dynamic SPD Encoder
# ---------------------------------------------------------------------------
from model_hop_masked_transformer_final_3 import (
    HopMaskedTransformerLayer,
    PostGATv2Block,
    build_head_hop_sets,
    _build_alternating_hop_sets,
)


class HopMaskedTransformerModelDynamicSPD(nn.Module):
    """Hop-Masked Transformer with Dynamic Max-Diameter Padded Shortest Path Sequence Encodings."""

    def __init__(
        self,
        hidden_dim: int = 120,
        num_heads: int = 12,
        ffn_ratio: float = 1.0,
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
        num_post_gat_layers: int = 0,
        num_gat_heads: int = 4,
        cross_hop_hop_embedding: bool = False,
        cross_hop_no_ffn: bool = False,
        blend_adj_power: bool = False,
        use_edge_bias: bool = False,
        use_rrwp: bool = False,
        # Dynamic SPD settings
        use_dynamic_spd_path: bool = False,
        use_dynamic_spd_edge: bool = False,
        spd_num_paths: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_hops = max_hops
        self.task_level = task_level
        self.graph_pool = graph_pool
        self.use_alternating = (hop_mode == "alternating")
        self.use_dynamic_spd_path = use_dynamic_spd_path
        self.use_dynamic_spd_edge = use_dynamic_spd_edge

        self.node_encoder = build_node_encoder(
            hidden_dim,
            dataset_name=dataset_name,
            lap_pe_dim=lap_pe_dim,
            node_feat_dim=node_feat_dim,
        )

        self.spd_node_encoder = None
        if use_dynamic_spd_path:
            self.spd_node_encoder = SPDPathDynamicNodeEncoder(
                hidden_dim=hidden_dim,
                num_paths=spd_num_paths,
                dropout=dropout,
            )

        self.spd_edge_encoder = None
        if use_dynamic_spd_edge:
            self.spd_edge_encoder = SPDPathDynamicEdgeEncoder(
                hidden_dim=hidden_dim,
                num_paths=spd_num_paths,
                dropout=dropout,
                dataset_name=dataset_name,
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
                max_hops=max_hops,
                cross_hop_hop_embedding=cross_hop_hop_embedding,
                cross_hop_no_ffn=cross_hop_no_ffn,
                blend_adj_power=blend_adj_power,
                use_edge_bias=use_edge_bias,
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

        # 1. Dynamic Padded Shortest Path Node Sequence Enrichment
        if self.use_dynamic_spd_path and spd_paths is not None and self.spd_node_encoder is not None:
            dense_x = self.spd_node_encoder(dense_x, spd_paths, spd_dists, node_mask=node_mask)

        # 2. Dynamic Padded Shortest Path Edge Sequence Enrichment
        if self.use_dynamic_spd_edge and spd_paths is not None and self.spd_edge_encoder is not None:
            dense_x = self.spd_edge_encoder(dense_x, spd_paths, batch, node_mask=node_mask)

        # Pass through Transformer Stack with per-head masks
        for i, layer in enumerate(self.layers):
            hop_sets = self.per_layer_hop_sets[i] if self.use_alternating else None
            per_head_mask = self._build_per_head_mask(dist_masks, hop_sets=hop_sets)
            dense_x = layer(
                dense_x,
                per_head_mask,
                node_mask=node_mask,
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
