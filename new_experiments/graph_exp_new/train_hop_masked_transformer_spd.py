# train_hop_masked_transformer_spd.py
"""
Standalone training script for Hop-Masked Transformer with Shortest Path Geodesic Embeddings.

Supports 3 Separate Shortest Path Formulations:
1. --use_spd_path      : Deterministic Shortest Path Node Sequence Enrichment
2. --use_spd_edge_path : Deterministic Shortest Path Edge Sequence Enrichment
3. --use_spd_bias      : Continuous Geodesic Metric Attention Bias via Floyd-Warshall Embedding

Zero disturbance to any existing training scripts, models, or running jobs.
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from data import DATASET_CHOICES, get_loaders
from metrics import build_task, compute_pos_weight
from model_hop_masked_transformer_spd import (
    HopMaskedTransformerModelSPD,
    set_attn_diagnostics,
)
from optim_utils import build_grouped_optimizer_and_scheduler

os.environ["PYTHON_HASH_SEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_parser():
    p = argparse.ArgumentParser(description="Train SPD Geodesic Hop-Masked Transformer")

    # Dataset
    p.add_argument("--dataset", type=str, default="PascalVOC-SP", choices=DATASET_CHOICES)
    p.add_argument("--max_hops", type=int, default=12)
    p.add_argument("--subgraph_mode", type=str, default="partition", choices=["partition", "egonet"])
    p.add_argument("--num_parts", type=int, default=128)
    p.add_argument("--egonet_hops", type=int, default=2)
    p.add_argument("--egonet_max_nodes", type=int, default=1024)
    p.add_argument("--max_egonet_samples", type=int, default=None)
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)
    p.add_argument("--mask_type", type=str, default="shortest_path")
    p.add_argument("--adj_self_loops", action="store_true", default=False)

    # Model Architecture
    p.add_argument("--hidden_dim", type=int, default=120)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--ffn_ratio", type=float, default=1.0)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--graph_pool", type=str, default="sum", choices=["sum", "mean", "attention"])
    p.add_argument("--norm_type", type=str, default="graph", choices=["layer", "rms", "graph"])
    p.add_argument("--v_head_dim", type=int, default=None)
    p.add_argument("--block_diag_out", action="store_true", default=True)
    p.add_argument("--dynamic_cross_hop", action="store_true", default=True)
    p.add_argument("--cross_hop_hop_embedding", action="store_true", default=False)
    p.add_argument("--multihop_attn", action="store_true", default=False)
    p.add_argument("--multihop_readout", type=str, default="mean", choices=["sum", "mean"])
    p.add_argument("--multihop_no_global", action="store_true", default=False)
    p.add_argument("--use_edge_features", action="store_true", default=False)
    p.add_argument("--use_pos_weight", action="store_true", default=False)
    p.add_argument("--focal_gamma", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--use_virtual_node", action="store_true", default=False)
    p.add_argument("--num_post_gat_layers", type=int, default=0)
    p.add_argument("--num_gat_heads", type=int, default=4)
    p.add_argument("--cross_hop_no_ffn", action="store_true", default=False)
    p.add_argument("--blend_adj_power", action="store_true", default=False)
    p.add_argument("--use_edge_bias", action="store_true", default=False)
    p.add_argument("--use_rrwp", action="store_true", default=False)
    p.add_argument("--rrwp_dim", type=int, default=8)

    # 3 SPD Formulations
    p.add_argument("--use_spd_bias", action="store_true", default=False,
                   help="Continuous Geodesic Metric Attention Bias via Floyd-Warshall embedding.")
    p.add_argument("--use_spd_path", action="store_true", default=False,
                   help="Deterministic Shortest Path Node Sequence Enrichment.")
    p.add_argument("--use_spd_edge_path", action="store_true", default=False,
                   help="Deterministic Shortest Path Edge Sequence Enrichment.")
    p.add_argument("--spd_num_paths", type=int, default=20)
    p.add_argument("--spd_path_len", type=int, default=3)
    p.add_argument("--spd_merge", type=str, default="concat", choices=["concat", "conv"])

    # Hop mode
    p.add_argument("--hop_mode", type=str, default="single")
    p.add_argument("--hop_window", type=int, default=1)
    p.add_argument("--num_global_heads", type=int, default=1)
    p.add_argument("--hop_file", type=str, default=None)

    # Optimization
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0003)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--reduce_lr_patience", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dist_mask_workers", type=int, default=8)
    p.add_argument("--log_head_stats_interval", type=int, default=10)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_spd")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None)
    return p


def parse_args():
    args, unknown = build_parser().parse_known_args()
    print(args.__dict__)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _move_batch_to_device(batch, device):
    if len(batch) == 5:
        pyg_batch, dist_masks, node_masks, spd_paths, spd_dists = batch
        return (pyg_batch.to(device), dist_masks.to(device), node_masks.to(device),
                spd_paths.to(device), spd_dists.to(device))
    pyg_batch, dist_masks, node_masks = batch
    return (pyg_batch.to(device), dist_masks.to(device), node_masks.to(device), None, None)


def train_one_epoch(model, loader, optimizer, scheduler, task, device, grad_clip=5.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    for raw_batch in loader:
        pyg_batch, dist_masks, node_masks, spd_paths, spd_dists = _move_batch_to_device(raw_batch, device)
        optimizer.zero_grad()

        logits = model(
            pyg_batch,
            dist_masks,
            node_mask=node_masks,
            spd_paths=spd_paths,
            spd_dists=spd_dists,
        )

        task_loss = task.loss(logits, pyg_batch.y)
        task_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(task_loss.item())
        n_batches += 1

        all_preds.append(task.predict(logits))
        all_targets.append(task.labels_to_numpy(pyg_batch.y))

    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        metric_val = task.compute_metric(y_pred, y_true)
    else:
        metric_val = 0.0
    return total_loss / max(n_batches, 1), float(metric_val)


@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    for raw_batch in loader:
        pyg_batch, dist_masks, node_masks, spd_paths, spd_dists = _move_batch_to_device(raw_batch, device)
        logits = model(
            pyg_batch,
            dist_masks,
            node_mask=node_masks,
            spd_paths=spd_paths,
            spd_dists=spd_dists,
        )
        task_loss = task.loss(logits, pyg_batch.y)
        total_loss += float(task_loss.item())
        n_batches += 1

        all_preds.append(task.predict(logits))
        all_targets.append(task.labels_to_numpy(pyg_batch.y))

    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        metric_val = task.compute_metric(y_pred, y_true)
    else:
        metric_val = 0.0
    return total_loss / max(n_batches, 1), float(metric_val)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Use MERW/SPD loader when sequence paths are enabled
    use_path_loader = args.use_spd_path or args.use_spd_edge_path

    train_loader, val_loader, test_loader, _, _, _, dataset_info = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dist_masks=True,
        max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe,
        lap_pe_dim=args.lap_pe_dim,
        dataset_name=args.dataset,
        return_info=True,
        subgraph_mode=args.subgraph_mode,
        num_parts=args.num_parts,
        egonet_hops=args.egonet_hops,
        egonet_max_nodes=args.egonet_max_nodes,
        max_egonet_samples=args.max_egonet_samples,
        seed=args.seed,
        use_merw=use_path_loader,
        merw_num_paths=args.spd_num_paths,
        merw_path_len=args.spd_path_len,
    )
    dataset_name = dataset_info["name"]

    pos_weight = None
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(train_loader).to(args.device)

    task = build_task(
        dataset_name,
        dataset_info=dataset_info,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    model = HopMaskedTransformerModelSPD(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_hops=args.max_hops,
        hop_mode=args.hop_mode,
        hop_window=args.hop_window,
        hop_file=args.hop_file,
        num_global_heads=args.num_global_heads,
        output_dim=dataset_info["output_dim"],
        graph_pool=args.graph_pool,
        task_level=dataset_info["level"],
        dataset_name=args.dataset,
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        node_feat_dim=dataset_info.get("node_feat_dim"),
        block_diag_out=args.block_diag_out,
        dynamic_cross_hop=args.dynamic_cross_hop,
        norm_type=args.norm_type,
        v_head_dim=args.v_head_dim,
        mask_type=args.mask_type,
        adj_self_loops=args.adj_self_loops,
        use_virtual_node=args.use_virtual_node,
        num_post_gat_layers=args.num_post_gat_layers,
        num_gat_heads=args.num_gat_heads,
        cross_hop_hop_embedding=args.cross_hop_hop_embedding,
        use_edge_features=args.use_edge_features,
        edge_feat_dim=dataset_info.get("edge_feat_dim"),
        cross_hop_no_ffn=args.cross_hop_no_ffn,
        blend_adj_power=args.blend_adj_power,
        use_edge_bias=args.use_edge_bias,
        use_rrwp=args.use_rrwp,
        rrwp_dim=args.rrwp_dim,
        # SPD settings
        use_spd_bias=args.use_spd_bias,
        use_spd_path=args.use_spd_path,
        use_spd_edge_path=args.use_spd_edge_path,
        spd_num_paths=args.spd_num_paths,
        spd_path_len=args.spd_path_len,
        spd_merge=args.spd_merge,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.3f}M", flush=True)

    total_steps = max(args.max_epochs * len(train_loader), 1)
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=list(model.named_parameters()),
        lr_max=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    best_val_metric = -float("inf")
    best_epoch = 0
    test_at_best_val = 0.0
    patience_counter = 0

    metric_name = dataset_info["metric_name"]
    metric_label = "AP" if "ap" in metric_name else ("F1" if "f1" in metric_name else "Score")

    for epoch in range(args.max_epochs):
        t0 = time.time()
        train_loss, train_metric = train_one_epoch(
            model, train_loader, optimizer, scheduler, task, args.device, grad_clip=args.grad_clip
        )
        val_loss, val_metric = evaluate(model, val_loader, task, args.device)
        test_loss, test_metric = evaluate(model, test_loader, task, args.device)
        dt = time.time() - t0

        improved = val_metric > best_val_metric
        if improved:
            best_val_metric = val_metric
            best_epoch = epoch
            test_at_best_val = test_metric
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metric": val_metric,
                    "test_metric": test_metric,
                    "args": vars(args),
                },
                os.path.join(args.save_dir, f"best_{args.dataset}.pt"),
            )
        else:
            patience_counter += 1

        print(
            f"epoch {epoch:03d} | {dt:5.1f}s | "
            f"train loss {train_loss:.4f} {metric_label} {train_metric:.4f} | "
            f"val loss {val_loss:.4f} {metric_label} {val_metric:.4f} | "
            f"test loss {test_loss:.4f} {metric_label} {test_metric:.4f} | "
            f"best val {best_val_metric:.4f} (epoch {best_epoch}, test {test_at_best_val})",
            flush=True,
        )

        diag = model.get_path_diagnostics()
        if diag is not None:
            gnorm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gnorm += float(p.grad.norm(2).item()) ** 2
            gnorm = gnorm ** 0.5
            print(
                f"[epoch {epoch:03d}] SPD Diagnostics: "
                f"attn_ent={diag['entropy']:.3f} part={diag['participation']:.2f}/{args.spd_num_paths} "
                f"top_path={diag['top_path']:.3f} gate_alpha={diag['gate']:.4f} grad_norm={gnorm:.4f}",
                flush=True,
            )

        if patience_counter >= args.patience:
            print(f"early stopping at epoch {epoch:03d} (no val improvement in {args.patience} epochs)", flush=True)
            break

    print(
        f"BEST: val {metric_label} {best_val_metric:.4f} | test {metric_label} {test_at_best_val:.4f} (epoch {best_epoch})",
        flush=True,
    )


if __name__ == "__main__":
    main()
