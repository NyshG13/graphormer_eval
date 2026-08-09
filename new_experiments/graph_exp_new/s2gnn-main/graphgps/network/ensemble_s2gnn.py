import torch
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.layer import new_layer_config
from torch_geometric.utils import to_dense_batch
import torch_geometric.graphgym.register as register
from functools import partial
import numpy as np

from graphgps.layer.s2_message_passing import FeatureBatchGNNLayer, GCNConvGNNLayer, GATConvGNNLayer, GatedGCNConvGNNLayer, GNNLayer
from graphgps.layer.s2_spectral import FeatureBatchSpectralLayer, MLPMultiBatch
from graphgps.layer.chebnet_conv_layer import ChebNetIILayer
from graphgps.network.s2gnn import BatchS2GNNGNNLayer

from graphgps.layer.hop_masked_transformer_layer import (
    build_head_hop_sets,
    HopMaskedTransformerLayer,
)

@register_network('ensemble_s2gnn')
class EnsembleS2GNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # ---- 1. Independent Feature Encoders ----
        self.encoder_a = FeatureEncoder(cfg.gnn.dim_inner)
        self.encoder_b = FeatureEncoder(cfg.gnn.dim_inner)
        dim_in = self.encoder_a.dim_in
        
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp_a = MLPMultiBatch(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            self.pre_mp_b = MLPMultiBatch(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        # ---- 2. Branch A: S2GNN (Spatial + Spectral) ----
        self.branch_a = self._build_s2gnn_branch(dim_in)

        # ---- 3. Branch B: Hop-Masked Transformer ----
        self.branch_b = self._build_hop_masked_branch(dim_in)

        # ---- 4. Task Head (Fusion) ----
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = cfg.gnn.layers_mp <= 0
        # The head will receive concatenated features: dim_inner * 2
        self.post_mp = GNNHead(cfg.gnn.dim_inner * 2, dim_out, is_first)

    def _build_s2gnn_branch(self, dim_in):
        # Build spat/spec prototypes
        def build_spatial_layer(model_type, make_undirected, use_edge_attr, adj_norm, dir_aggr):
            if model_type == 'none' or model_type is None:
                return None
            elif model_type == 'lin_gnn':
                return partial(FeatureBatchGNNLayer,
                               make_undirected=make_undirected,
                               use_edge_attr=use_edge_attr,
                               normalize=adj_norm,
                               dir_aggr=dir_aggr)
            elif model_type == 'gcnconv':
                return partial(GCNConvGNNLayer,
                               make_undirected=make_undirected,
                               use_edge_attr=use_edge_attr,
                               normalize=adj_norm)
            elif model_type == 'gatconv':
                return GATConvGNNLayer
            elif model_type == 'gatedgcnconv':
                return GatedGCNConvGNNLayer
            elif model_type.startswith('chebconv'):
                kwargs = {}
                if '-' in model_type:
                    kwargs['K'] = int(model_type.split('-')[-1])
                return partial(ChebNetIILayer, **kwargs)
            else:
                return GNNLayer

        def build_spectral_layer():
            if not cfg.posenc_MagLapPE.enable:
                return None
            else:
                return FeatureBatchSpectralLayer

        spat_model = build_spatial_layer(cfg.gnn.layer_type,
                                         cfg.gnn.make_undirected,
                                         cfg.gnn.use_edge_attr,
                                         cfg.gnn.adj_norm,
                                         cfg.gnn.dir_aggr)
        spec_model = build_spectral_layer()
        layers = []

        spat_layer_skip = [i % cfg.gnn.layers_mp for i in cfg.gnn.layer_skip if i < cfg.gnn.layers_mp]
        spec_layer_skip = [i % cfg.gnn.layers_mp for i in cfg.gnn.spectral.layer_skip if i < cfg.gnn.layers_mp]

        for i in range(cfg.gnn.layers_mp):
            is_first, is_last = (i == 0), (i == cfg.gnn.layers_mp - 1)
            dim_in_ = dim_in if is_first else cfg.gnn.dim_inner
            dim_out_ = cfg.gnn.dim_inner
            
            layer_cfg = new_layer_config(
                dim_in_, dim_out_, cfg.gnn.spectral.filter_layers,
                has_act=True, has_bias=True, cfg=cfg)

            if cfg.gnn.spectral.combine_with_spatial:
                with_node_residual = (cfg.gnn.spectral.combine_with_spatial == 'mamba_like')
                spat_layer = spat_model(
                    layer_cfg, is_first=is_first, is_last=is_last,
                    overwrite_x=False, with_node_residual=with_node_residual)
                spec_layer = spec_model(
                    layer_cfg, is_first=is_first, overwrite_x=False,
                    with_node_residual=with_node_residual)
                layers.append(BatchS2GNNGNNLayer(
                    layer_cfg, spat_layer, spec_layer,
                    with_node_residual=cfg.gnn.residual,
                    aggr_mode=cfg.gnn.spectral.combine_with_spatial,
                    norm=cfg.gnn.spectral.combine_with_spatial_norm))

        return nn.ModuleList(layers)

    def _build_hop_masked_branch(self, dim_in):
        hm = cfg.gnn.hop_masked
        hidden_dim = hm.hidden_dim
        num_heads = hm.num_heads
        num_hops = hm.num_hops
        num_tf_layers = hm.num_layers
        ffn_ratio = hm.ffn_ratio
        
        self.head_hop_sets = build_head_hop_sets(
            max_hops=num_hops,
            num_heads=num_heads,
            mode="single",
            window=0,
            include_self=True,
            num_global_heads=1,
        )
        self.max_hops = num_hops
        
        tf_layers = nn.ModuleList([
            HopMaskedTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * ffn_ratio,
                dropout=cfg.gnn.dropout,
                norm_type="layer",
            )
            for _ in range(num_tf_layers)
        ])
        
        return tf_layers

    def _build_per_head_mask(self, dist_masks: torch.Tensor) -> torch.Tensor:
        B, K_runtime, N, _ = dist_masks.shape
        H = len(self.head_hop_sets)
        out = dist_masks.new_zeros(B, H, N, N, dtype=torch.bool)
        for h, hop_set in enumerate(self.head_hop_sets):
            if hop_set is None:
                out[:, h] = True
                continue
            idx = [k for k in hop_set if k < K_runtime]
            if not idx:
                continue
            stacked = dist_masks[:, idx].bool().any(dim=1)
            out[:, h] = stacked
        return out

    def _pad_dist_masks(self, batch, N_max: int, device: torch.device) -> torch.Tensor:
        dist_mask_list = batch.dist_mask
        B = batch.num_graphs
        K = self.max_hops

        dm_padded = torch.zeros(B, K, N_max, N_max, dtype=torch.float32, device=device)
        for i in range(B):
            dm_i = dist_mask_list[i]
            if isinstance(dm_i, np.ndarray):
                dm_i = torch.from_numpy(dm_i.astype(np.float32))
            dm_i = dm_i.to(device)
            K_i, N_i, _ = dm_i.shape
            K_use = min(K_i, K)
            dm_padded[i, :K_use, :N_i, :N_i] = dm_i[:K_use]
        return dm_padded

    def forward(self, batch):
        if not hasattr(batch, 'num_graphs'):
            batch.num_graphs = 1

        # 1. Independent Encoders
        batch_a = batch.clone()
        batch_a = self.encoder_a(batch_a)
        if hasattr(self, 'pre_mp_a'):
            batch_a = self.pre_mp_a(batch_a)
            
        batch_b = batch.clone()
        batch_b = self.encoder_b(batch_b)
        if hasattr(self, 'pre_mp_b'):
            batch_b = self.pre_mp_b(batch_b)
            
        # 2. Branch A: S2GNN
        for layer in self.branch_a:
            batch_a = layer(batch_a)
        x_a = batch_a.x
        
        # 3. Branch B: Hop-Masked Transformer
        x_in_b = batch_b.x
        dense_x, mask = to_dense_batch(x_in_b, batch_b.batch)
        B, N_max, d = dense_x.shape
        
        dist_masks = self._pad_dist_masks(batch, N_max, dense_x.device)
        per_head_mask = self._build_per_head_mask(dist_masks)
        
        h = dense_x
        for t_layer in self.branch_b:
            h = t_layer(h, per_head_mask, mask)
        
        x_b = h[mask]
        
        # 4. Concatenate and pass to fusion head
        batch.x = torch.cat([x_a, x_b], dim=-1)
        batch = self.post_mp(batch)
        
        return batch
