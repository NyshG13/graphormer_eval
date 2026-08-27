"""
Diagnostic script: Log mask density and attention entropy per layer/head.

Usage:
  # With a trained checkpoint (recommended for meaningful entropy):
  python run_diagnostics.py --dataset_name cora --checkpoint_path ./experiments/.../training_checkpoints/best_model.pt

  # Without checkpoint (random init — still shows mask density correctly):
  python run_diagnostics.py --dataset_name cora

  # Run across multiple datasets:
  python run_diagnostics.py --dataset_name cora --checkpoint_path ./path/to/cora_model.pt
  python run_diagnostics.py --dataset_name citeseer --checkpoint_path ./path/to/citeseer_model.pt
"""

import torch
import argparse
import json
import os
import sys
import math
import numpy as np

from graphormer_hf.modeling_graphormer import (
    GraphormerForNodeClassification,
    enable_attention_diagnostics,
)
from graphormer_hf.configuration_graphormer import GraphormerConfig
from graphormer_hf.collating_graphormer import GraphormerDataCollator
import dataset_utils

import wandb
import dotenv
dotenv.load_dotenv()


def compute_diagnostics_single_pass(model, loader, device, config, split="train"):
    """Run a single forward pass and collect diagnostics from all attention layers."""
    model.eval()
    
    # Collect diagnostics manually (without wandb, printed to stdout)
    diagnostics = {}
    
    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    batch[k] = [i.to(device) for i in batch[k]]
            
            labels = batch["labels"]
            mask_name = f"{split}_mask"
            node_mask = getattr(loader.dataset[0], mask_name, None)
            if node_mask is not None:
                node_mask = node_mask.view(-1).to(device) & ~torch.isnan(labels.view(-1)).to(device)
            else:
                node_mask = torch.ones(labels.shape, dtype=torch.int32, device=device)

            # Forward pass (triggers diagnostics logging to wandb)
            outputs = model(**batch, node_mask=node_mask, log_step=1, log_group=split)
            
            # Also manually extract mask & entropy info from each layer
            encoder = model.encoder.graph_encoder
            for layer_idx, layer in enumerate(encoder.layers):
                attn = layer.self_attn
                if hasattr(attn, 'last_attn_logits') and attn.last_attn_logits is not None:
                    logits = attn.last_attn_logits  # [bsz*heads, tgt, src]
                    bsz = logits.shape[0] // attn.num_heads
                    tgt_len = logits.shape[1]
                    src_len = logits.shape[2]
                    
                    # Mask density
                    is_masked = torch.isinf(logits) & (logits < 0)
                    pct_active = (1.0 - is_masked.sum().item() / logits.numel()) * 100
                    
                    # Per-head
                    is_masked_per_head = is_masked.view(bsz, attn.num_heads, tgt_len, src_len)
                    head_actives = []
                    for h in range(attn.num_heads):
                        hm = is_masked_per_head[:, h, :, :]
                        head_actives.append((1.0 - hm.sum().item() / hm.numel()) * 100)
                    
                    # Entropy
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    p = probs.clamp(min=1e-12)
                    row_entropy = -(p * p.log()).sum(dim=-1)  # [bsz*heads, tgt]
                    max_entropy = math.log(src_len)
                    
                    row_entropy_per_head = row_entropy.view(bsz, attn.num_heads, tgt_len)
                    head_entropies = []
                    for h in range(attn.num_heads):
                        head_entropies.append(row_entropy_per_head[:, h, :].mean().item())
                    
                    diagnostics[f"layer_{layer_idx}"] = {
                        "pct_active_overall": round(pct_active, 4),
                        "per_head_pct_active": [round(x, 4) for x in head_actives],
                        "mean_entropy": round(row_entropy.mean().item(), 4),
                        "max_possible_entropy": round(max_entropy, 4),
                        "normalized_entropy": round(row_entropy.mean().item() / max_entropy, 4) if max_entropy > 0 else 0,
                        "per_head_entropy": [round(x, 4) for x in head_entropies],
                        "per_head_normalized_entropy": [round(x / max_entropy, 4) if max_entropy > 0 else 0 for x in head_entropies],
                        "num_nodes": src_len,
                    }
            
            break  # Only need one batch (single graph for node classification)
    
    return diagnostics


def print_diagnostics(diagnostics, dataset_name):
    print(f"\n{'='*80}")
    print(f"  ATTENTION DIAGNOSTICS — {dataset_name.upper()}")
    print(f"{'='*80}")
    
    for layer_name, d in sorted(diagnostics.items()):
        print(f"\n--- {layer_name} (num_nodes={d['num_nodes']}) ---")
        print(f"  Mask density (% active positions): {d['pct_active_overall']:.2f}%")
        print(f"  Mean attention entropy: {d['mean_entropy']:.4f} / {d['max_possible_entropy']:.4f}")
        print(f"  Normalized entropy (0=focused, 1=uniform): {d['normalized_entropy']:.4f}")
        
        print(f"\n  Per-head breakdown:")
        print(f"  {'Head':>6} | {'% Active':>10} | {'Entropy':>10} | {'Norm Entropy':>14}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}")
        for h in range(len(d['per_head_pct_active'])):
            print(f"  {h:>6} | {d['per_head_pct_active'][h]:>9.2f}% | {d['per_head_entropy'][h]:>10.4f} | {d['per_head_normalized_entropy'][h]:>14.4f}")
    
    print(f"\n{'='*80}")
    

def main():
    parser = argparse.ArgumentParser(description="Attention Diagnostics")
    parser.add_argument("--dataset_name", type=str, default="cora", help="Dataset name")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--spatial_pos_max", type=int, default=20, help="spatial_pos_max used during training")
    parser.add_argument("--edge_type", type=str, default="multi_hop")
    parser.add_argument("--remove_attn_bias", action="store_true")
    parser.add_argument("--enable_spatial_encoder", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name (optional, logs to wandb if set)")
    args = parser.parse_args()

    dataset_classes = {
        "cora": 7,
        "citeseer": 6,
        "pubmed": 3,
        "film": 5,
        "deezer": 6,
        "ogbn-arxiv": 40,
        "ogbn-products": 47,
    }

    # Build config matching training config
    config = GraphormerConfig(
        num_hidden_layers=6,
        embedding_dim=768 // 4,
        ffn_embedding_dim=768 // 4,
        num_attention_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_classes=dataset_classes[args.dataset_name],
        edge_type=args.edge_type,
        enable_spatial_encoder=args.enable_spatial_encoder,
        remove_attn_bias=args.remove_attn_bias,
        dataset_name=args.dataset_name,
        spatial_pos_max=args.spatial_pos_max,
        # Defaults for unused features
        enable_diffusion=False,
        enable_layerwise_diffusion=False,
        node_augmentation=False,
        augment_edges=False,
        create_subgraph=False,
        diffusion_steps=50,
        experiment_dir="./diagnostics_output",
    )

    os.makedirs("./diagnostics_output", exist_ok=True)

    # Init wandb if requested
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"diagnostics_{args.dataset_name}",
            config=vars(args),
        )
    else:
        wandb.init(mode="disabled")

    # Load data
    train_loader, _, _ = dataset_utils.load_data(args.dataset_name, config=config)

    # Build model
    model = GraphormerForNodeClassification(config)
    
    # Load checkpoint if provided
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint_path:
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print("Checkpoint loaded.")
    else:
        print("No checkpoint provided — running with random initialization.")
        print("Mask density will be accurate; entropy values will NOT reflect a trained model.\n")

    model.to(device)
    
    # Enable diagnostics on all attention layers
    enable_attention_diagnostics(model, enabled=True, log_interval=1)
    
    # Run diagnostics
    diagnostics = compute_diagnostics_single_pass(model, train_loader, device, config, split="train")
    
    # Print results
    print_diagnostics(diagnostics, args.dataset_name)
    
    # Save to JSON
    output_path = f"./diagnostics_output/{args.dataset_name}_diagnostics.json"
    with open(output_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Also log summary to wandb
    if args.wandb_project:
        wandb.log({"diagnostics_summary": diagnostics})
    
    wandb.finish()


if __name__ == "__main__":
    main()
