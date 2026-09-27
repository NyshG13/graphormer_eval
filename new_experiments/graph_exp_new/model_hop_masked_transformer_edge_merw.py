# model_hop_masked_transformer_edge_merw.py
"""
Hop-Masked Transformer with Edge-based MERW (Maximal Entropy Random Walk) Path Enrichment.

Instead of gathering node features along the random walk, this module extracts the
sequence of edge/bond embeddings along each sampled walk:
    Path p = [v0 -> v1 -> v2 -> ... -> vL]
    Edges  = [e1=(v0,v1), e2=(v1,v2), ..., eL=(vL-1, vL)]
The edge sequence (path_len * d) is merged into 1 vector per path (1 * d),
and target node v0 attends over its M candidate edge-paths to enrich its representation.
"""

from __future__ import annotations

import json
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
# Edge-based MERW Path Encoder
# ---------------------------------------------------------------------------
class MERWPathEdgeEncoder(nn.Module):
    """Encodes the sequence of edges along sampled MERW walks to enrich node features."""

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
        self.path_len = path_len       # Number of edges in walk (e.g. 3 for 4-node path)
        self.num_paths = num_paths     # Number of sampled paths per node (e.g. 20)
        self.merge_mode = merge_mode
        self.scale = hidden_dim ** -0.5

        # 1. Edge encoder (embeds raw edge_attr to hidden_dim)
        self.bond_encoder = build_bond_encoder(
            hidden_dim, dataset_name=dataset_name, edge_feat_dim=edge_feat_dim
        )

        # 2. Step positional embedding for edge transitions (step 1, step 2, ... step L)
        self.step_embed = nn.Embedding(self.path_len + 1, hidden_dim)

        # 3. Path-merging mechanism (maps L * d -> d)
        if merge_mode == "concat":
            self.path_mlp = nn.Sequential(
                nn.Linear(self.path_len * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif merge_mode == "conv":
            self.conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2),
            )
        elif merge_mode == "cross_attn":
            self.local_q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.local_k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.local_v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.local_drop = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown path merge mode: {merge_mode}")

        self.path_norm = nn.LayerNorm(hidden_dim)

        # 4. Path-to-Node Attention Aggregator
        self.node_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Learnable residual gate initialized to 0.0 (smoothly blends path features into base node)
        self.path_gate = nn.Parameter(torch.tensor(0.0))
        self._diag = None

    def forward(
        self,
        dense_x: torch.Tensor,     # (B, N, d) base node embeddings
        merw_paths: torch.Tensor,  # (B, N, M, path_len + 1) node indices in walk
        batch: torch_geometric.data.Batch,
        node_mask: Optional[torch.Tensor] = None, # (B, N) bool
    ) -> torch.Tensor:
        B, N, d = dense_x.shape
        M, L = self.num_paths, self.path_len

        # If no edge attributes in graph, return node embeddings as fallback
        if not hasattr(batch, "edge_attr") or batch.edge_attr is None or batch.edge_index is None:
            return dense_x

        # 1. Embed all graph edges to dimension d
        edge_attr = batch.edge_attr
        edge_emb = self.bond_encoder(edge_attr)  # (total_E, d)

        # 2. Scatter into a dense adjacency feature tensor E_dense: (B, N, N, d)
        E_dense = torch.zeros(B, N, N, d, device=dense_x.device, dtype=dense_x.dtype)
        src_global, dst_global = batch.edge_index[0], batch.edge_index[1]
        edge_batch = batch.batch[src_global]

        ptr = getattr(batch, "ptr", None)
        if ptr is not None:
            src_local = src_global - ptr[edge_batch]
            dst_local = dst_global - ptr[edge_batch]
        else:
            node_offsets = torch.zeros(B, dtype=torch.long, device=dense_x.device)
            for b in range(B):
                node_offsets[b] = (batch.batch < b).sum()
            src_local = src_global - node_offsets[edge_batch]
            dst_local = dst_global - node_offsets[edge_batch]

        valid_edges = (src_local < N) & (dst_local < N) & (src_local >= 0) & (dst_local >= 0)
        E_dense[edge_batch[valid_edges], src_local[valid_edges], dst_local[valid_edges]] = edge_emb[valid_edges]

        # 3. Look up the L edges for all M sampled walks: [v0->v1, v1->v2, ..., vL-1->vL]
        clamped_paths = merw_paths.clamp(min=0, max=N - 1)  # (B, N, M, L+1)
        src_nodes = clamped_paths[:, :, :, :-1]             # (B, N, M, L)
        dst_nodes = clamped_paths[:, :, :, 1:]              # (B, N, M, L)

        b_idx = torch.arange(B, device=dense_x.device).view(B, 1, 1, 1).expand(B, N, M, L)
        edge_tokens = E_dense[b_idx, src_nodes, dst_nodes]  # (B, N, M, L, d)

        # 4. Add step positional embeddings for transitions 0..L-1
        step_idx = torch.arange(L, device=dense_x.device).view(1, 1, 1, L)
        step_pos = self.step_embed(step_idx)
        edge_tokens = edge_tokens + step_pos                # (B, N, M, L, d)

        # 5. Merge the L edge tokens into 1 vector per path (B, N, M, d)
        if self.merge_mode == "concat":
            flat_tokens = edge_tokens.view(B, N, M, L * d)
            h_paths = self.path_mlp(flat_tokens)            # (B, N, M, d)
        elif self.merge_mode == "conv":
            conv_in = edge_tokens.view(B * N * M, L, d).transpose(1, 2)
            conv_out = self.conv(conv_in)[:, :, :L]
            h_paths = conv_out[:, :, -1].view(B, N, M, d)
        elif self.merge_mode == "cross_attn":
            q_node_local = self.local_q_proj(dense_x).unsqueeze(2).unsqueeze(3) # (B, N, 1, 1, d)
            k_edge = self.local_k_proj(edge_tokens)          # (B, N, M, L, d)
            v_edge = self.local_v_proj(edge_tokens)          # (B, N, M, L, d)
            local_scores = torch.matmul(q_node_local, k_edge.transpose(-2, -1)) * self.scale # (B, N, M, 1, L)
            local_attn = F.softmax(local_scores, dim=-1)
            local_attn = self.local_drop(local_attn)
            h_paths = torch.matmul(local_attn, v_edge).squeeze(-2) # (B, N, M, d)

        h_paths = self.path_norm(h_paths)

        # 6. Target node attends over its M candidate edge-path vectors
        q_node = self.node_q_proj(dense_x).unsqueeze(2)     # (B, N, 1, d)
        k_path = self.path_k_proj(h_paths)                  # (B, N, M, d)
        v_path = self.path_v_proj(h_paths)                  # (B, N, M, d)

        scores = torch.matmul(q_node, k_path.transpose(-2, -1)) * self.scale # (B, N, 1, M)
        raw_attn = F.softmax(scores, dim=-1)                # (B, N, 1, M)
        attn_weights = self.attn_drop(raw_attn)

        # Diagnostics stash
        with torch.no_grad():
            p = raw_attn.squeeze(2).clamp_min(1e-12)        # (B, N, M)
            ent = -(p * p.log()).sum(-1)                    # (B, N)
            part = 1.0 / p.pow(2).sum(-1).clamp_min(1e-12)  # (B, N)
            valid = node_mask if node_mask is not None else torch.ones(B, N, dtype=torch.bool, device=dense_x.device)
            if valid.any():
                ent_val = float(ent[valid].mean().cpu())
                part_val = float(part[valid].mean().cpu())
                top_p_val = float(p.max(dim=-1)[0][valid].mean().cpu())
            else:
                ent_val = float(ent.mean().cpu())
                part_val = float(part.mean().cpu())
                top_p_val = float(p.max(dim=-1)[0].mean().cpu())
            self._diag = {
                "entropy": ent_val,
                "participation": part_val,
                "top_path": top_p_val,
                "gate": float(self.path_gate.detach().cpu()),
            }

        x_edge_path = torch.matmul(attn_weights, v_path).squeeze(2) # (B, N, d)

        # Base node residual + gated edge-path representation
        out = dense_x + self.path_gate * self.final_norm(x_edge_path)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Import HopMaskedTransformer building blocks from model_hop_masked_transformer_final_3
# ---------------------------------------------------------------------------
from model_hop_masked_transformer_final_3 import (
    HopMaskedTransformerLayer,
    PostGATv2Block,
    build_head_hop_sets,
    _build_alternating_hop_sets,
    compute_gate_aux_loss,
    set_attn_diagnostics,
)


class HopMaskedTransformerModelEdgeMERW(nn.Module):
    """Hop-masked multi-head transformer with Edge-based MERW path enrichment."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 8,
        ffn_ratio: float = 1,
        num_layers: int = 1,
        dropout: float = 0.2,
        max_hops: int = 40,
        hop_mode: str = "contiguous",
        hop_window: int = 1,
        hop_file: Optional[str] = None,
        num_global_heads: int = 0,
        output_dim: int = 10,
        graph_pool: str = "sum",
        task_level: str = "graph",
        dataset_name: str = "PascalVOC-SP",
        node_feat_dim: Optional[int] = None,
        lap_pe_dim: int = 0,
        block_diag_out: bool = False,
        dynamic_cross_hop: bool = False,
        norm_type: str = "layer",
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
        multihop_attn: bool = False,
        multihop_readout: str = "sum",
        multihop_include_global: bool = True,
        embed_dropout: float = 0.0,
        use_merw: bool = True,
        merw_num_paths: int = 20,
        merw_path_len: int = 3,
        merw_merge: str = "concat",
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_hops = max_hops
        self.task_level = task_level
        self.graph_pool = graph_pool
        self.mask_type = mask_type
        self.adj_self_loops = adj_self_loops
        self.use_moe_gating = use_moe_gating
        self.multihop_attn = multihop_attn
        self.balance_coeff = balance_coeff
        self.entropy_coeff = entropy_coeff
        self.use_virtual_node = use_virtual_node
        self.use_alternating = (hop_mode == "alternating")
        self.blend_adj_power = blend_adj_power
        self.use_edge_bias = use_edge_bias
        self.use_rrwp = use_rrwp
        self.rrwp_dim = rrwp_dim
        self.use_merw = use_merw

        # Instantiate Edge-based MERW Encoder
        self.merw_path_encoder = None
        if use_merw:
            self.merw_path_encoder = MERWPathEdgeEncoder(
                hidden_dim=hidden_dim,
                path_len=merw_path_len,
                num_paths=merw_num_paths,
                merge_mode=merw_merge,
                dropout=dropout,
                dataset_name=dataset_name,
                edge_feat_dim=edge_feat_dim,
            )

        self.embed_drop = nn.Dropout(embed_dropout)
        self.dropout = dropout

        # Per-layer hop sets
        if self.use_alternating:
            even_sets, odd_sets = _build_alternating_hop_sets(
                max_hops=max_hops, num_heads=num_heads, include_self=True, num_global_heads=num_global_heads
            )
            self.per_layer_hop_sets = [even_sets if (i % 2 == 0) else odd_sets for i in range(num_layers)]
            self.head_hop_sets = even_sets
        else:
            self.head_hop_sets = build_head_hop_sets(
                max_hops=max_hops, num_heads=num_heads, mode=hop_mode, window=hop_window,
                include_self=True, num_global_heads=num_global_heads, hop_file=hop_file
            )
            self.per_layer_hop_sets = [self.head_hop_sets] * num_layers

        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            dataset_name=dataset_name,
            node_feat_dim=node_feat_dim,
            use_edge_features=use_edge_features,
            edge_feat_dim=edge_feat_dim,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            HopMaskedTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                max_hops=max_hops,
                block_diag_out=block_diag_out,
                dynamic_cross_hop=dynamic_cross_hop,
                norm_type=norm_type,
                v_head_dim=v_head_dim,
                use_moe_gating=use_moe_gating,
                top_k=top_k,
                gate_noise=gate_noise,
                hop_membership=None,
                cross_hop_no_ffn=cross_hop_no_ffn,
                blend_adj_power=blend_adj_power,
                use_edge_bias=use_edge_bias,
                use_rrwp=use_rrwp,
                multihop_attn=multihop_attn,
                multihop_readout=multihop_readout,
                multihop_include_global=multihop_include_global,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Head
        if task_level == "graph":
            self.head = nn.Linear(hidden_dim, output_dim)
        else:
            self.head = nn.Linear(hidden_dim, output_dim)

    def encode_dense(self, batch):
        h = self.encoder(batch.x, batch.edge_index, getattr(batch, "edge_attr", None), getattr(batch, "lap_pe", None))
        dense_x, dense_mask = to_dense_batch(h, batch.batch)
        return dense_x, dense_mask

    def forward(self, batch, dist_masks, node_masks=None, merw_paths=None, merw_dists=None, return_gate_weights=False):
        dense_x, dense_mask = self.encode_dense(batch)

        # Apply Edge-based MERW enrichment
        if self.use_merw and self.merw_path_encoder is not None and merw_paths is not None:
            dense_x = self.merw_path_encoder(dense_x, merw_paths, batch, dense_mask)

        dense_x = self.embed_drop(dense_x)
        nm = node_masks if node_masks is not None else dense_mask
        mask_source = dist_masks

        x = dense_x
        all_gate_weights = []

        if self.multihop_attn:
            for layer in self.layers:
                x = layer(x, mask_source, nm)
            aux_loss = x.new_tensor(0.0)
            _exported_gw = None
        else:
            for layer_idx, layer in enumerate(self.layers):
                per_head_mask = torch.ones(x.shape[0], self.num_heads, x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
                for h_idx, hs in enumerate(self.per_layer_hop_sets[layer_idx]):
                    if hs is not None:
                        m = torch.zeros_like(mask_source[:, 0], dtype=torch.bool)
                        for k in hs:
                            if k < mask_source.shape[1]:
                                m = m | (mask_source[:, k] > 0)
                        per_head_mask[:, h_idx] = m
                x = layer(x, per_head_mask, nm)
            aux_loss = x.new_tensor(0.0)
            _exported_gw = None

        x = self.norm(x)

        if self.task_level == "graph":
            if self.graph_pool == "mean":
                mask_f = nm.unsqueeze(-1).float()
                graph_emb = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            else:
                graph_emb = (x * nm.unsqueeze(-1)).sum(dim=1)
            logits = self.head(graph_emb)
            node_emb = x
        else:
            logits = self.head(x)
            node_emb = x

        return logits, node_emb, aux_loss, _exported_gw
