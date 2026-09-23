"""Hop-Masked S2GNN — Experiment 4: Path-Aware Attention.

Registered as ``hop_masked_s2gnn_exp4``.

This is the most architecturally novel experiment.  For each node pair (i,j),
the attention VALUE is augmented with the projected features of the
intermediate node m on the shortest path i → m → ... → j:

    V_effective[i,j] = V[j]  +  β · W_path(x[m])

This propagates actual intermediate node information through the path —
not path statistics — which is the key idea the TA highlighted from PathNet.

The number of heads that use path augmentation is controlled by the new
config key  cfg.gnn.hop_masked.num_path_heads  (default: 4).

Changes vs baseline
-------------------
- Uses PathAwareTransformerLayer.
- Loads path_features and passes path_mid (padded) to the transformer.
- Requires the path-aware DataLoader / collate_path_features.

Config additions
----------------
In your YAML add under gnn.hop_masked:
    num_path_heads: 4   # how many restricted heads use path augmentation
"""

from __future__ import annotations

import math
from typing import Optional

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
from graphgps.layer.hop_masked_transformer_exp4 import PathAwareTransformerLayer
from graphgps.layer.s2_spectral import FeatureBatchSpectralLayer
from graphgps.loader.path_mask_utils import pad_path_features
from graphgps.network.hop_masked_s2gnn import (
    BatchHopMaskedS2GNNLayer, _IdentitySpectralLayer,
)
from graphgps.network.hop_masked_s2gnn_exp1 import _dist_int_from_masks


class BatchHopMaskedS2GNNLayerExp4(BatchHopMaskedS2GNNLayer):
    """BatchHopMaskedS2GNNLayer with path-aware attention (Exp 4)."""

    def forward(self, batch):
        from torch_geometric.graphgym.config import cfg
        x_in = batch.x
        branch_ablation = None if self.training else self.branch_ablation

        dense_x, mask = to_dense_batch(x_in, batch.batch)
        B, N_max, d  = dense_x.shape
        dist_masks    = self._pad_dist_masks(batch, N_max, dense_x.device)
        per_head_mask = self._build_per_head_mask(dist_masks)
        dist_int      = _dist_int_from_masks(dist_masks, N_max)

        # Pad path features — need path_mid (global intermediate node indices)
        pf       = pad_path_features(batch, N_max, self.max_hops, dense_x.device)
        path_mid = pf['path_mid']   # (B, N_max, N_max) long, -1 = none

        if cfg.gnn.spectral.combine_with_spatial is None:
            # ---- SEQUENTIAL ----
            h = dense_x
            for t_layer in self.transformer_layers:
                h = t_layer(h, per_head_mask, mask, dist_int, path_mid)

            spat_out = h[mask]
            if branch_ablation == 'spatial':
                spat_out = torch.zeros_like(spat_out)

            batch.x = x_in + spat_out if self.with_node_residual else spat_out
            x_after_spat = batch.x

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
            spec_out = self.spec_layer(batch)

            h = dense_x
            for t_layer in self.transformer_layers:
                h = t_layer(h, per_head_mask, mask, dist_int, path_mid)

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


@register_network('hop_masked_s2gnn_exp4')
class HopMaskedS2GNNExp4(nn.Module):
    """S2GNN with Path-Aware Attention — Experiment 4."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        hm = cfg.gnn.hop_masked
        hidden_dim    = hm.hidden_dim
        num_heads     = hm.num_heads
        num_hops      = hm.num_hops
        num_tf_layers = hm.num_layers
        ffn_ratio     = hm.ffn_ratio
        # New config key — defaults to 4 if not set
        num_path_heads = getattr(hm, 'num_path_heads', 4)

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
                PathAwareTransformerLayer(hidden_dim=hidden_dim,
                                          num_heads=num_heads,
                                          ffn_dim=ffn_dim,
                                          max_hops=num_hops,
                                          num_path_heads=num_path_heads,
                                          dropout=dropout,
                                          norm_type="layer")
                for _ in range(num_tf_layers)
            ])

            layers.append(BatchHopMaskedS2GNNLayerExp4(
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
