"""Two-phase boosting: S2GNN (Branch A) + HopMaskedS2GNN (Branch B).

Phase 1 — Train Branch A (S2GNN) to convergence on original targets.
Phase 2 — Freeze encoder + Branch A.  Train Branch B (HopMaskedS2GNN)
           on the *residual* errors left by Branch A.

Final prediction (eval): pred_a + pred_b

This implements true gradient-boosting semantics:
  - Model 1 is trained first to convergence
  - Model 2 is trained on the errors of Model 1
  - The final prediction is the sum of both models
"""

import logging

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
    """Two-phase boosting ensemble with a shared FeatureEncoder.

    Architecture::

        Phase 1 (train Branch A only):
            Shared FeatureEncoder -> Branch A (S2GNN) -> post_mp_a -> pred_a
            Loss = L(pred_a, true)

        Phase 2 (freeze A, train Branch B on residuals):
            Shared FeatureEncoder (frozen) -> batch_encoded
                |                                  |
                v                                  v
            Branch A (frozen) -> pred_a     Branch B (HopMasked) -> pred_b
                                            Loss = L(pred_b, true - pred_a)

        Eval (both phases):
            Phase 1: pred_a
            Phase 2: pred_a + pred_b
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

        # Independent prediction heads
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = cfg.gnn.layers_mp <= 0
        self.post_mp_a = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)
        self.post_mp_b = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)

        # Boosting phase tracking (1 = train A, 2 = freeze A + train B)
        self.boosting_phase = 1

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------
    def enter_phase2(self):
        """Freeze encoder + Branch A + its head.  Unfreeze Branch B + head.

        Called by the training loop after phase1_epochs have completed.
        """
        logging.info("[Boosting] === Entering Phase 2 ===")
        logging.info("[Boosting] Freezing: encoder, branch_a, post_mp_a")
        logging.info("[Boosting] Training: branch_b, post_mp_b")

        self.boosting_phase = 2

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Freeze Branch A + its head
        for p in self.branch_a.parameters():
            p.requires_grad = False
        for p in self.post_mp_a.parameters():
            p.requires_grad = False

        # Ensure Branch B + its head are trainable
        for p in self.branch_b.parameters():
            p.requires_grad = True
        for p in self.post_mp_b.parameters():
            p.requires_grad = True

        # Log parameter counts
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"[Boosting] Frozen params: {frozen:,}  |  "
                     f"Trainable params: {trainable:,}")

    def phase2_trainable_parameters(self):
        """Return only the trainable parameters for Phase 2 optimizer."""
        params = []
        for p in self.branch_b.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.post_mp_b.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_branch_a(self, batch):
        """Run encoder + Branch A + head_a.  Returns (pred_a, label, batch)."""
        batch = self.encoder(batch)

        batch_a = batch.clone()
        if hasattr(batch, 'dist_mask'):
            batch_a.dist_mask = batch.dist_mask
        batch_a = self.branch_a(batch_a)

        # Head A
        batch_a_copy = batch.clone()
        batch_a_copy.x = batch_a.x
        out_a = self.post_mp_a(batch_a_copy)

        if isinstance(out_a, tuple):
            pred_a = out_a[0]
            label = out_a[1]
        else:
            pred_a = out_a
            label = None

        return pred_a, label, batch

    def _run_branch_b(self, batch):
        """Run Branch B + head_b on an already-encoded batch.  Returns pred_b."""
        batch_b = batch.clone()
        if hasattr(batch, 'dist_mask'):
            batch_b.dist_mask = batch.dist_mask
        batch_b = self.branch_b(batch_b)

        # Head B
        batch_b_copy = batch.clone()
        batch_b_copy.x = batch_b.x
        out_b = self.post_mp_b(batch_b_copy)

        if isinstance(out_b, tuple):
            pred_b = out_b[0]
        else:
            pred_b = out_b

        return pred_b

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch):
        # Set num_graphs if not available
        if not hasattr(batch, 'num_graphs'):
            batch.num_graphs = 1

        if self.boosting_phase == 1:
            # --- PHASE 1: Train Branch A only ---
            pred_a, label, _ = self._run_branch_a(batch)
            # Return single prediction (no tuple wrapping needed)
            return pred_a, label

        else:
            # --- PHASE 2: Freeze A, train B on residuals ---
            pred_a, label, encoded_batch = self._run_branch_a(batch)

            # Detach pred_a so no gradients flow through Branch A
            pred_a = pred_a.detach()

            # Run Branch B on the same encoded features
            pred_b = self._run_branch_b(encoded_batch)

            # Return the combined prediction and original label.
            # During training, the native loss function (e.g. BCE or L1) will 
            # automatically backpropagate through pred_b to correct pred_a's mistakes.
            pred_final = pred_a + pred_b
            return pred_final, label
