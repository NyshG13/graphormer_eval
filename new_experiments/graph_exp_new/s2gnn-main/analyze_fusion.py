import sys
import torch
import numpy as np

def analyze_fusion_weights(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # PyTorch Lightning wraps the model in a module, usually 'model.' prefix
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # We are looking for the first linear layer in the GNNHead (post_mp)
    # In MLPGraphHead, it's typically self.mlp[1] (index 0 is Dropout)
    # The key might look like 'model.post_mp.mlp.1.weight'
    weight_key = None
    for key in state_dict.keys():
        if 'post_mp' in key and 'mlp.1.weight' in key:
            weight_key = key
            break
            
    if weight_key is None:
        print("Could not find the fusion linear layer weights in the checkpoint.")
        print("Available keys related to post_mp:")
        for key in state_dict.keys():
            if 'post_mp' in key:
                print(f"  - {key}")
        return
        
    print(f"\nFound fusion layer weights: {weight_key}")
    W = state_dict[weight_key] # Shape should be (480, 480) for Ensemble
    
    out_features, in_features = W.shape
    print(f"Weight matrix shape: {out_features} x {in_features}")
    
    if in_features % 2 != 0:
        print(f"Warning: Expected an even number of input features for a concatenated ensemble, got {in_features}.")
        return
        
    half = in_features // 2
    
    # Split the weights corresponding to Branch A (S2GNN) and Branch B (HopMasked)
    W_A = W[:, :half]
    W_B = W[:, half:]
    
    # Calculate Frobenius norms (overall magnitude of weights)
    norm_A = torch.linalg.norm(W_A).item()
    norm_B = torch.linalg.norm(W_B).item()
    
    # Calculate L1 norms (absolute sum of weights)
    l1_A = torch.sum(torch.abs(W_A)).item()
    l1_B = torch.sum(torch.abs(W_B)).item()
    
    print("\n--- Fusion Level Analysis ---")
    print(f"S2GNN branch (first {half} dims):")
    print(f"  L2 (Frobenius) Norm: {norm_A:.4f}")
    print(f"  L1 (Absolute) Norm:  {l1_A:.4f}")
    
    print(f"\nHopMasked branch (last {half} dims):")
    print(f"  L2 (Frobenius) Norm: {norm_B:.4f}")
    print(f"  L1 (Absolute) Norm:  {l1_B:.4f}")
    
    total_l2 = norm_A + norm_B
    total_l1 = l1_A + l1_B
    
    print("\n--- Relative Importance ---")
    print(f"Based on L2 Norms:")
    print(f"  S2GNN:     {(norm_A / total_l2 * 100):.2f}%")
    print(f"  HopMasked: {(norm_B / total_l2 * 100):.2f}%")
    
    print(f"Based on L1 Norms:")
    print(f"  S2GNN:     {(l1_A / total_l1 * 100):.2f}%")
    print(f"  HopMasked: {(l1_B / total_l1 * 100):.2f}%")
    
    if norm_A > norm_B * 1.5:
        print("\nConclusion: The model heavily focuses on the S2GNN (local) features.")
    elif norm_B > norm_A * 1.5:
        print("\nConclusion: The model heavily focuses on the HopMasked (global) features.")
    else:
        print("\nConclusion: The model blends both local and global features relatively evenly.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_fusion.py <path_to_checkpoint.ckpt>")
    else:
        analyze_fusion_weights(sys.argv[1])
