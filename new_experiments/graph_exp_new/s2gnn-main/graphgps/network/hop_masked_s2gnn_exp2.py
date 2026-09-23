"""Hop-Masked S2GNN — Experiment 2: Shortest-Path Spectral Adjacency.

Registered as ``hop_masked_s2gnn_exp2``.

Change vs baseline
------------------
The SPECTRAL branch uses a shortest-path-filtered and re-normalized adjacency
instead of the standard GCN-normalized A.

An edge (i,j) in the original graph is retained in A_sp only if it lies on
at least one shortest path between some pair of nodes.  A simple sufficient
condition: for every edge (i,j), check if there exists some node k such that
d(i,k) = 1 + d(j,k), i.e., going through i shortens the path to k by 1.
If no such k exists, the edge is "wasted" — it doesn't help any shortest path
— and is dropped.

In practice we implement this using the precomputed integer distance matrix:
    A_sp[i,j] = 1  iff  A[i,j]=1  AND  min_k ( |d(i,k) - d(j,k)| ) == 1

After filtering, A_sp is symmetrized and GCN-normalized, then stored in
batch.adj_norm before the spectral layer runs (which caches adj_norm).

The Transformer is unchanged from the baseline.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.gnn import FeatureEncoder
from torch_geometric.utils import to_dense_batch
import torch_geometric.graphgym.register as register
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphgps.layer.hop_masked_transformer_layer import (
    build_head_hop_sets, HopMaskedTransformerLayer,
)
from graphgps.layer.s2_spectral import FeatureBatchSpectralLayer
from graphgps.network.hop_masked_s2gnn import (
    BatchHopMaskedS2GNNLayer,
    _IdentitySpectralLayer,
)
from graphgps.network.hop_masked_s2gnn_exp1 import _dist_int_from_masks


def _build_sp_adj_sparse(batch, dist_masks, N_max, device):
    """Build a shortest-path-filtered SparseTensor for the spectral branch.

    Uses the already-padded dist_masks (B, K, N_max, N_max) to compute,
    per graph, which edges survive the SP filter, then assembles a batched
    SparseTensor compatible with the spectral layer's adj_norm slot.

    Returns SparseTensor of shape (total_nodes, total_nodes).
    """
    ptr = batch.ptr.cpu()
    total_nodes = int(ptr[-1])
    edge_index_orig = batch.edge_index  # (2, E) — batched global indices

    # Build a mask over existing edges: keep edge (u,v) if it lies on
    # some shortest path.  Criterion: d(u,k) != d(v,k) for all k means
    # neither direction shortens; at least one k with |d(u,k)-d(v,k)|=1
    # means it does contribute.
    #
    # Efficient batch implementation: for each graph i, compute
    # sp_mask[i] = (dist_masks[i,1] & (any k: d(.,k) diff by 1)).
    # We use the dist_int derived from dist_masks for speed.

    dist_int = _dist_int_from_masks(dist_masks, N_max)  # (B, N_max, N_max)

    src_list, dst_list, val_list = [], [], []
    B = batch.num_graphs
    for b in range(B):
        n_start = int(ptr[b])
        n_end   = int(ptr[b + 1])
        n = n_end - n_start
        d = dist_int[b, :n, :n].cpu()  # (n, n) long, -1 = unreachable

        # 1-hop neighbours of every node
        adj1 = (d == 1).float()  # (n, n)

        # An edge (i,j) survives if there is some k where
        # d(i,k) >= 0 and d(j,k) >= 0 and |d(i,k) - d(j,k)| == 1.
        # Vectorised: for each pair (i,j) with adj1[i,j]=1,
        # check if any column k has abs(d[i,k]-d[j,k])==1.
        di = d.unsqueeze(1)   # (n, 1, n)
        dj = d.unsqueeze(0)   # (1, n, n)
        diff = (di - dj).abs()  # (n, n, n) — [i, j, k]
        valid_k = (di >= 0) & (dj >= 0)
        sp_contrib = ((diff == 1) & valid_k).any(dim=2)  # (n, n) bool
        sp_mask = adj1.bool() & sp_contrib

        # Gather surviving edge indices (local)
        rows, cols = sp_mask.nonzero(as_tuple=True)

        src_list.append(rows + n_start)
        dst_list.append(cols + n_start)
        val_list.append(torch.ones(rows.shape[0], dtype=torch.float32))

    if not src_list:
        # Degenerate: no edges survive — fall back to identity
        src = torch.arange(total_nodes)
        dst = torch.arange(total_nodes)
        vals = torch.ones(total_nodes)
    else:
        src  = torch.cat(src_list)
        dst  = torch.cat(dst_list)
        vals = torch.cat(val_list)

    ei  = torch.stack([src, dst], dim=0).to(device)
    wts = vals.to(device)

    # GCN normalize the SP adjacency
    ei, wts = gcn_norm(ei, wts, num_nodes=total_nodes,
                       add_self_loops=True, flow='source_to_target')
    adj_sp = SparseTensor.from_edge_index(
        ei, wts, (total_nodes, total_nodes)
    ).coalesce()
    return adj_sp


class BatchHopMaskedS2GNNLayerExp2(BatchHopMaskedS2GNNLayer):
    """BatchHopMaskedS2GNNLayer that injects SP-filtered adjacency (Exp 2)."""

    def forward(self, batch):
        from torch_geometric.graphgym.config import cfg
        x_in = batch.x
        branch_ablation = None if self.training else self.branch_ablation

        if cfg.gnn.spectral.combine_with_spatial is None:
            # ---- SEQUENTIAL ----
            dense_x, mask = to_dense_batch(x_in, batch.batch)
            B, N_max, d  = dense_x.shape
            dist_masks    = self._pad_dist_masks(batch, N_max, dense_x.device)
            per_head_mask = self._build_per_head_mask(dist_masks)

            h = dense_x
            for t_layer in self.transformer_layers:
                h = t_layer(h, per_head_mask, mask)

            spat_out = h[mask]
            if branch_ablation == 'spatial':
                spat_out = torch.zeros_like(spat_out)

            batch.x = x_in + spat_out if self.with_node_residual else spat_out
            x_after_spat = batch.x

            # Inject SP adjacency into batch before spectral layer
            sp_adj = _build_sp_adj_sparse(batch, dist_masks, N_max, x_in.device)
            batch.adj_norm = sp_adj

            spec_out = self.spec_layer(batch)
            if branch_ablation == 'spectral':
                spec_out = torch.zeros_like(spec_out)

            batch.x = (x_after_spat + spec_out if self.with_node_residual
                       else spec_out)

            if len(self.transformer_layers) > 0:
                self._log_metrics(self.transformer_layers[0], dist_masks,
                                  spat_out, spec_out, mask, x_in, batch.x)
            return batch
        else:
            # ---- PARALLEL ----
            dense_x, mask = to_dense_batch(x_in, batch.batch)
            B, N_max, d  = dense_x.shape
            dist_masks    = self._pad_dist_masks(batch, N_max, dense_x.device)
            per_head_mask = self._build_per_head_mask(dist_masks)

            # Inject SP adjacency
            sp_adj = _build_sp_adj_sparse(batch, dist_masks, N_max, x_in.device)
            batch.adj_norm = sp_adj
            spec_out = self.spec_layer(batch)

            h = dense_x
            for t_layer in self.transformer_layers:
                h = t_layer(h, per_head_mask, mask)

            spat_out = h[mask]
            if branch_ablation == 'spatial':
                spat_out = torch.zeros_like(spat_out)

            y = spec_out + spat_out
            if branch_ablation == 'spectral':
                y = spat_out
            if self.with_node_residual:
                y = y + x_in
            batch.x = y / math.sqrt(self.norm_factor) if self.norm else y

            if len(self.transformer_layers) > 0:
                self._log_metrics(self.transformer_layers[0], dist_masks,
                                  spat_out, spec_out, mask, x_in, batch.x)
            return batch


@register_network('hop_masked_s2gnn_exp2')
class HopMaskedS2GNNExp2(nn.Module):
    """S2GNN with shortest-path spectral adjacency — Experiment 2.

    Transformer is unchanged; only the spectral adjacency is modified.
    Does NOT require path_features in the batch.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        hm = cfg.gnn.hop_masked
        hidden_dim    = hm.hidden_dim
        num_heads     = hm.num_heads
        num_hops      = hm.num_hops
        num_tf_layers = hm.num_layers
        ffn_ratio     = hm.ffn_ratio

        self.encoder = FeatureEncoder(cfg.gnn.dim_inner)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            from graphgps.layer.s2_spectral import MLPMultiBatch
            self.pre_mp = MLPMultiBatch(dim_in, cfg.gnn.dim_inner,
                                        cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        self.head_hop_sets = build_head_hop_sets(
            max_hops=num_hops, num_heads=num_heads, mode="single",
            window=0, include_self=True, num_global_heads=1,
        )
        self.max_hops = num_hops

        self.proj_in  = nn.Linear(cfg.gnn.dim_inner, hidden_dim) \
                        if cfg.gnn.dim_inner != hidden_dim else None
        self.proj_out = nn.Linear(hidden_dim, cfg.gnn.dim_inner) \
                        if hidden_dim != cfg.gnn.dim_inner else None

        dropout = cfg.gnn.dropout
        ffn_dim = hidden_dim * ffn_ratio
        spec_layer_skip = [i % cfg.gnn.layers_mp for i in cfg.gnn.spectral.layer_skip
                           if i < cfg.gnn.layers_mp]

        layers = []
        for i in range(cfg.gnn.layers_mp):
            is_first = (i == 0)
            spec_dim  = cfg.gnn.dim_inner
            layer_cfg = new_layer_config(spec_dim, spec_dim,
                                         cfg.gnn.spectral.filter_layers,
                                         has_act=True, has_bias=True, cfg=cfg)
            spec_layer = (FeatureBatchSpectralLayer(layer_cfg, is_first=is_first,
                                                    overwrite_x=False,
                                                    with_node_residual=False)
                          if i not in spec_layer_skip
                          else _IdentitySpectralLayer())

            tf_layers = nn.ModuleList([
                HopMaskedTransformerLayer(hidden_dim=hidden_dim,
                                          num_heads=num_heads,
                                          ffn_dim=ffn_dim,
                                          dropout=dropout,
                                          norm_type="layer")
                for _ in range(num_tf_layers)
            ])

            layers.append(BatchHopMaskedS2GNNLayerExp2(
                transformer_layers=tf_layers,
                spec_layer=spec_layer,
                head_hop_sets=self.head_hop_sets,
                max_hops=self.max_hops,
                with_node_residual=cfg.gnn.residual,
                norm=cfg.gnn.spectral.combine_with_spatial_norm
                     if cfg.gnn.spectral.combine_with_spatial is not None
                     else True,
            ))

        self.gnn_layers = nn.ModuleList(layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(cfg.gnn.dim_inner, dim_out,
                               cfg.gnn.layers_mp <= 0)

    def set_branch_ablation(self, branch=None):
        for layer in self.gnn_layers:
            if hasattr(layer, 'set_branch_ablation'):
                layer.set_branch_ablation(branch)

    def forward(self, batch):
        if not hasattr(batch, 'num_graphs'):
            batch.num_graphs = 1
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        if self.proj_in is not None:
            batch.x = self.proj_in(batch.x)
        for layer in self.gnn_layers:
            batch = layer(batch)
        if self.proj_out is not None:
            batch.x = self.proj_out(batch.x)
        batch = self.post_mp(batch)
        return batch
