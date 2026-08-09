import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head

@register_head('ensemble_graph')
class EnsembleGraphHead(nn.Module):
    """
    Fusion MLP prediction head for Ensemble S2GNN.
    
    Args:
        dim_in (int): Input dimension (typically 2 * dim_inner because of concatenation).
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        is_first (bool): Not used, kept for compatibility.
    """
    def __init__(self, dim_in, dim_out, is_first=False):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        
        # Fusion MLP
        dim_inner = cfg.gnn.dim_inner
        dropout = cfg.gnn.dropout
        layers = []
        
        # 1. Projection from concat_dim -> dim_inner
        layers.append(nn.Linear(dim_in, dim_inner, bias=True))
        layers.append(register.act_dict[cfg.gnn.act]())
        layers.append(nn.Dropout(dropout))
        
        # 2. Rest of MLP
        L = cfg.gnn.layers_post_mp
        for _ in range(L - 1):
            layers.append(nn.Linear(dim_inner, dim_inner, bias=True))
            layers.append(register.act_dict[cfg.gnn.act]())
            layers.append(nn.Dropout(dropout))
            
        layers.append(nn.Linear(dim_inner, dim_out, bias=True))
        
        self.mlp = nn.Sequential(*layers)

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        # Pool first (batch.x is concatenated embeddings from both branches)
        x = self.pooling_fun(batch.x, batch.batch)
        
        # Fusion and output
        y = self.mlp(x)
        
        batch.graph_feature = y
        pred, label = self._apply_index(batch)
        return pred, label
