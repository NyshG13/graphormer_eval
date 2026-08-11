import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graphgps.network.s2gnn import S2GNN
from graphgps.network.hop_masked_s2gnn import HopMaskedS2GNN

@register_network('sequential_s2gnn')
class SequentialS2GNN(nn.Module):
    """
    Sequential Architecture: S2GNN -> HopMaskedS2GNN
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # 1. S2GNN (Spatial + Spectral) acts as the local feature extractor
        self.s2gnn_branch = S2GNN(dim_in, dim_out)
        self.s2gnn_branch.post_mp = nn.Identity() # Strip head
        
        # 2. HopMaskedS2GNN acts as the global router on top of S2GNN
        self.hop_masked_branch = HopMaskedS2GNN(dim_in, dim_out)
        self.hop_masked_branch.encoder = nn.Identity() # Strip encoder (already encoded by S2GNN)
        if hasattr(self.hop_masked_branch, 'pre_mp'):
            self.hop_masked_branch.pre_mp = nn.Identity()
        self.hop_masked_branch.post_mp = nn.Identity() # Strip head
        
        # Unified Head
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = cfg.gnn.layers_mp <= 0
        self.post_mp = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)
        
    def forward(self, batch):
        # Local routing
        batch = self.s2gnn_branch(batch)
        
        # Global routing
        batch = self.hop_masked_branch(batch)
        
        # Task head
        batch = self.post_mp(batch)
        
        return batch
