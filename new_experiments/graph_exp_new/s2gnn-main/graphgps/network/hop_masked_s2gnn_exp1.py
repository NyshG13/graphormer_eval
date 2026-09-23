"""Hop-Masked S2GNN — Experiment 1: Learnable Hop Bias.

Registered as ``hop_masked_s2gnn_exp1``.

Change vs baseline
------------------
- Uses HopBiasTransformerLayer instead of HopMaskedTransformerLayer.
- Pads the integer distance matrix (dist_int) and passes it to every
  transformer sub-layer so the hop bias embedding can look up per-hop scalars.
- Requires batch.path_features (loaded by the path-aware DataLoader).
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

from graphgps.layer.hop_masked_transformer_layer import build_head_hop_sets
from graphgps.layer.hop_masked_transformer_exp1 import HopBiasTransformerLayer
from graphgps.layer.s2_spectral import FeatureBatchSpectralLayer
from graphgps.loader.path_mask_utils import pad_path_features

# Reuse helpers from the baseline network
from graphgps.network.hop_masked_s2gnn import (
    BatchHopMaskedS2GNNLayer,
    _IdentitySpectralLayer,
)


class BatchHopMaskedS2GNNLayerExp1(BatchHopMaskedS2GNNLayer):
    """BatchHopMaskedS2GNNLayer that uses HopBiasTransformerLayer (Exp 1).

    Overrides forward to: (a) build dist_int from the dist_masks data,
    (b) pass dist_int to each transformer sub-layer.
    """

    def forward(self, batch):
        from torch_geometric.graphgym.config import cfg
        x_in = batch.x
        branch_ablation = None if self.training else self.branch_ablation

        if cfg.gnn.spectral.combine_with_spatial is None:
            # ---- SEQUENTIAL (transformer → spectral) ----
            dense_x, mask = to_dense_batch(x_in, batch.batch)
            B, N_max, d  = dense_x.shape

            dist_masks = self._pad_dist_masks(batch, N_max, dense_x.device)
            per_head_mask = self._build_per_head_mask(dist_masks)

            # Build integer distance matrix from dist_masks
            dist_int = _dist_int_from_masks(dist_masks, N_max)

            h = dense_x
            for t_layer in self.transformer_layers:
                h = t_layer(h, per_head_mask, mask, dist_int)

            spat_out = h[mask]
            if branch_ablation == 'spatial':
                spat_out = torch.zeros_like(spat_out)

            if self.with_node_residual:
                batch.x = x_in + spat_out
            else:
                batch.x = spat_out
            x_after_spat = batch.x

            spec_out = self.spec_layer(batch)
            if branch_ablation == 'spectral':
                spec_out = torch.zeros_like(spec_out)

            batch.x = x_after_spat + spec_out if self.with_node_residual else spec_out

            if len(self.transformer_layers) > 0:
                self._log_metrics(self.transformer_layers[0], dist_masks,
                                  spat_out, spec_out, mask, x_in, batch.x)
            return batch

        else:
            # ---- PARALLEL ----
            spec_out = self.spec_layer(batch)

            dense_x, mask = to_dense_batch(x_in, batch.batch)
            B, N_max, d  = dense_x.shape
            dist_masks    = self._pad_dist_masks(batch, N_max, dense_x.device)
            per_head_mask = self._build_per_head_mask(dist_masks)
            dist_int      = _dist_int_from_masks(dist_masks, N_max)

            h = dense_x
            for t_layer in self.transformer_layers:
                h = t_layer(h, per_head_mask, mask, dist_int)

            spat_out = h[mask]
            if branch_ablation == 'spatial':
                spat_out = torch.zeros_like(spat_out)

            y = spec_out + spat_out
            if branch_ablation == 'spectral':
                y = spat_out
            if self.with_node_residual:
                y = y + x_in
            if self.norm:
                batch.x = y / math.sqrt(self.norm_factor)
            else:
                batch.x = y

            if len(self.transformer_layers) > 0:
                self._log_metrics(self.transformer_layers[0], dist_masks,
                                  spat_out, spec_out, mask, x_in, batch.x)
            return batch


def _dist_int_from_masks(dist_masks: torch.Tensor, N_max: int) -> torch.Tensor:
    """Convert (B, K, N, N) bool masks → (B, N, N) long distance matrix.

    Result: value k where dist_masks[:,k] is True, -1 where no hop matches.
    """
    B, K, N, _ = dist_masks.shape
    dist_int = torch.full((B, N, N), -1, dtype=torch.long,
                          device=dist_masks.device)
    # Set in reverse so lower hops overwrite higher (lower = correct dist)
    for k in range(K - 1, -1, -1):
        dist_int = torch.where(dist_masks[:, k].bool(),
                               torch.full_like(dist_int, k),
                               dist_int)
    return dist_int


@register_network('hop_masked_s2gnn_exp1')
class HopMaskedS2GNNExp1(nn.Module):
    """S2GNN with Hop-Bias Transformer — Experiment 1."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        hm = cfg.gnn.hop_masked
        hidden_dim   = hm.hidden_dim
        num_heads    = hm.num_heads
        num_hops     = hm.num_hops
        num_tf_layers = hm.num_layers
        ffn_ratio    = hm.ffn_ratio

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
            spec_dim = cfg.gnn.dim_inner
            layer_cfg = new_layer_config(spec_dim, spec_dim,
                                         cfg.gnn.spectral.filter_layers,
                                         has_act=True, has_bias=True, cfg=cfg)
            spec_layer = (FeatureBatchSpectralLayer(layer_cfg, is_first=is_first,
                                                    overwrite_x=False,
                                                    with_node_residual=False)
                          if i not in spec_layer_skip
                          else _IdentitySpectralLayer())

            tf_layers = nn.ModuleList([
                HopBiasTransformerLayer(hidden_dim=hidden_dim,
                                        num_heads=num_heads,
                                        ffn_dim=ffn_dim,
                                        max_hops=num_hops,
                                        dropout=dropout,
                                        norm_type="layer")
                for _ in range(num_tf_layers)
            ])

            layers.append(BatchHopMaskedS2GNNLayerExp1(
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
