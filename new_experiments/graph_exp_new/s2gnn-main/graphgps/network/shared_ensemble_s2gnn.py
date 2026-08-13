"""Shared-encoder ensemble: S2GNN + HopMaskedS2GNN with one FeatureEncoder.

This variant uses a single shared FeatureEncoder for both branches,
reducing parameter count and memory usage while keeping the two
distinct processing branches (GCN-spatial vs Hop-Masked Transformer).

This serves as an ablation against the independent-encoder ensemble
(ensemble_s2gnn.py) to test whether independent encoders matter.
"""

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.gnn import FeatureEncoder

from graphgps.network.s2gnn import S2GNN
from graphgps.network.hop_masked_s2gnn import HopMaskedS2GNN


@register_network('shared_ensemble_s2gnn')
class SharedEnsembleS2GNN(nn.Module):
    """Parallel ensemble with a SHARED FeatureEncoder.

    Architecture::

        Shared FeatureEncoder -> batch_encoded
            |                        |
            v                        v
        Branch A (S2GNN)      Branch B (HopMaskedS2GNN)
            |                        |
            v                        v
        emb_A                    emb_B
            |                        |
            +--- concat(emb_A, emb_B) -> FusionHead -> prediction

    Unlike ``EnsembleS2GNN``, both branches share a single encoder.
    The internal encoders of each branch are replaced with ``nn.Identity``.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        # Shared encoder (same as what S2GNN/HopMaskedS2GNN use internally)
        self.encoder = FeatureEncoder(cfg.gnn.dim_inner)
        encoder_dim_in = self.encoder.dim_in  # output dim after encoding

        # Branch A: S2GNN (strip its encoder and head)
        self.branch_a = S2GNN(dim_in, dim_out)
        self.branch_a.encoder = nn.Identity()
        self.branch_a.post_mp = nn.Identity()

        # Branch B: HopMaskedS2GNN (strip its encoder and head)
        self.branch_b = HopMaskedS2GNN(dim_in, dim_out)
        self.branch_b.encoder = nn.Identity()
        self.branch_b.post_mp = nn.Identity()

        # Unified Ensemble Head (Modified for Boosting)
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = cfg.gnn.layers_mp <= 0
        self.post_mp_a = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)
        self.post_mp_b = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)

    def forward(self, batch):
        # Set num_graphs if not available
        if not hasattr(batch, 'num_graphs'):
            batch.num_graphs = 1

        # 1. Shared encoding (run once, both branches see the same features)
        batch = self.encoder(batch)

        # 2. Clone for independent branch processing
        batch_a = batch.clone()
        batch_b = batch.clone()

        # Ensure dist_mask (python list) is passed through clones
        if hasattr(batch, 'dist_mask'):
            batch_a.dist_mask = batch.dist_mask
            batch_b.dist_mask = batch.dist_mask

        # 3. Run both branches (their .encoder is Identity, so they skip encoding)
        batch_a = self.branch_a(batch_a)
        batch_b = self.branch_b(batch_b)

        # 4. Independent Predictions
        # S2GNN Head
        batch_a_copy = batch.clone()
        batch_a_copy.x = batch_a.x
        out_a = self.post_mp_a(batch_a_copy)
        
        # Hop-Masked Head
        batch_b_copy = batch.clone()
        batch_b_copy.x = batch_b.x
        out_b = self.post_mp_b(batch_b_copy)
        
        # Extract predictions (handling different return tuple lengths)
        if isinstance(out_a, tuple):
            pred_a = out_a[0]
            label = out_a[1]
        else:
            pred_a = out_a
            label = None
            
        if isinstance(out_b, tuple):
            pred_b = out_b[0]
        else:
            pred_b = out_b
            
        pred_final = pred_a + pred_b
        
        if self.training:
            return (pred_a, pred_final), label
        else:
            if isinstance(out_a, tuple) and len(out_a) > 2:
                return pred_final, label, out_a[2]
            return pred_final, label
