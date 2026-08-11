import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graphgps.network.s2gnn import S2GNN
from graphgps.network.hop_masked_s2gnn import HopMaskedS2GNN

@register_network('interleaved_s2gnn')
class InterleavedS2GNN(nn.Module):
    """
    Interleaved Architecture: [S2GNN Layer -> HopMasked Layer] x N
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Instantiate both to steal their layers
        s2gnn = S2GNN(dim_in, dim_out)
        hop_masked = HopMaskedS2GNN(dim_in, dim_out)
        
        # We keep the encoder and pre_mp from S2GNN
        self.encoder = s2gnn.encoder
        if hasattr(s2gnn, 'pre_mp'):
            self.pre_mp = s2gnn.pre_mp
            
        # Projections from HopMasked (in case inner dim != hidden dim)
        self.proj_in = hop_masked.proj_in
        self.proj_out = hop_masked.proj_out
        
        # Extract layers
        self.s2gnn_layers = s2gnn.gnn_layers
        self.hop_masked_layers = hop_masked.gnn_layers
        self.num_layers = cfg.gnn.layers_mp
        
        # Unified Head
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = cfg.gnn.layers_mp <= 0
        self.post_mp = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)
        
    def forward(self, batch):
        if not hasattr(batch, 'num_graphs'):
            batch.num_graphs = 1
            
        # Encode
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
            
        # Interleaved Message Passing
        for i in range(self.num_layers):
            # 1. S2GNN layer (local)
            batch = self.s2gnn_layers[i](batch)
            
            # 2. HopMasked layer (global)
            if self.proj_in is not None:
                batch.x = self.proj_in(batch.x)
                
            batch = self.hop_masked_layers[i](batch)
            
            if self.proj_out is not None:
                batch.x = self.proj_out(batch.x)
                
        # Task head
        batch = self.post_mp(batch)
        
        return batch
