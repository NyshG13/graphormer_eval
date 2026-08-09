import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graphgps.network.s2gnn import S2GNN
from graphgps.network.hop_masked_s2gnn import HopMaskedS2GNN

@register_network('ensemble_s2gnn')
class EnsembleS2GNN(nn.Module):
    """
    Parallel ensemble: S2GNN branch + Hop-Masked S2GNN branch.
    
    Architecture:
        Branch A: S2GNN (Spatial GCN + Spectral) -> emb_A
        Branch B: HopMaskedS2GNN (HopMasked Transformer + Spectral) -> emb_B
        
        concat(emb_A, emb_B) -> FusionHead -> prediction
        
    Since the 500K parameter constraint is lifted, we instantiate both branches 
    independently (independent feature encoders, independent layers) to allow them 
    to learn fully separated, specialized representations.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Branch A: Original S2GNN
        self.branch_a = S2GNN(dim_in, dim_out)
        # We strip the task head from Branch A, as we'll pool and classify later
        self.branch_a.post_mp = nn.Identity()
        
        # Branch B: Hop-Masked S2GNN
        self.branch_b = HopMaskedS2GNN(dim_in, dim_out)
        # We strip the task head from Branch B
        self.branch_b.post_mp = nn.Identity()

        # Both branches return node embeddings of size `cfg.gnn.dim_inner`
        self.concat_dim = cfg.gnn.dim_inner * 2
        
        # Unified Ensemble Head
        # It takes the concatenated node embeddings and applies the registered head (e.g., mlp_graph)
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = cfg.gnn.layers_mp <= 0
        self.post_mp = GNNHead(self.concat_dim, dim_out, is_first)
        
    def forward(self, batch):
        # Shallow copy the batch so each branch can independently modify batch.x
        batch_a = batch.clone()
        batch_b = batch.clone()
        
        # batch.clone() might not deepcopy custom python lists attached to the batch,
        # so we explicitly pass them over.
        if hasattr(batch, 'dist_mask'):
            batch_a.dist_mask = batch.dist_mask
            batch_b.dist_mask = batch.dist_mask
            
        # Run both branches to get their respective node embeddings
        batch_a = self.branch_a(batch_a)
        batch_b = self.branch_b(batch_b)
        
        # Concatenate node embeddings from both branches along the feature dimension
        batch.x = torch.cat([batch_a.x, batch_b.x], dim=-1)
        
        # Pass the concatenated node features to the unified head 
        # (which will handle graph pooling and MLP)
        batch = self.post_mp(batch)
        
        return batch
