# train_hop_masked_transformer_edge_merw.py
"""
Standalone training script for Hop-Masked Transformer with Edge-based MERW Path Enrichment.

Leaves all existing training scripts and models untouched.
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
from model_hop_masked_transformer_edge_merw import (
    HopMaskedTransformerModelEdgeMERW,
    set_attn_diagnostics,
)
from optim_utils import build_grouped_optimizer_and_scheduler

os.environ["PYTHON_HASH_SEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_parser():
    p = argparse.ArgumentParser(description="Train edge-merw hop-masked transformer")

    # Dataset
    p.add_argument("--dataset", type=str, default="PascalVOC-SP", choices=DATASET_CHOICES)
    p.add_argument("--max_hops", type=int, default=35)
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
    p.add_argument("--ffn_ratio", type=float, default=1)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--graph_pool", type=str, default="sum", choices=["sum", "mean", "attention"])
    p.add_argument("--norm_type", type=str, default="layer", choices=["layer", "rms", "graph"])
    p.add_argument("--v_head_dim", type=int, default=None)
    p.add_argument("--block_diag_out", action="store_true", default=False)
    p.add_argument("--dynamic_cross_hop", action="store_true", default=False)
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

    # Edge MERW Path Settings
    p.add_argument("--use_merw", action="store_true", default=True)
    p.add_argument("--merw_num_paths", type=int, default=20)
    p.add_argument("--merw_path_len", type=int, default=3)
    p.add_argument("--merw_merge", type=str, default="concat", choices=["concat", "conv", "cross_attn"])

    # Hop mode
    p.add_argument("--hop_mode", type=str, default="contiguous")
    p.add_argument("--hop_window", type=int, default=1)
    p.add_argument("--num_global_heads", type=int, default=0)
    p.add_argument("--hop_file", type=str, default=None)

    # MoE gating
    p.add_argument("--use_moe_gating", action="store_true", default=False)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--gate_noise", type=float, default=0.1)
    p.add_argument("--balance_coeff", type=float, default=0.01)
    p.add_argument("--entropy_coeff", type=float, default=0.01)

    # Optimization
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0003)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--reduce_lr_patience", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dist_mask_workers", type=int, default=8)
    p.add_argument("--log_head_stats_interval", type=int, default=10)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_edge_merw")
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
        pyg_batch, dist_masks, node_masks, merw_paths, merw_dists = batch
        return (pyg_batch.to(device), dist_masks.to(device), node_masks.to(device),
                merw_paths.to(device), merw_dists.to(device))
    pyg_batch, dist_masks, node_masks = batch
    return pyg_batch.to(device), dist_masks.to(device), node_masks.to(device), None, None


def run_epoch(model, loader, task, device, optimizer=None, scheduler=None, grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)

    losses, preds_acc, labels_acc = [], [], []
    for batch in loader:
        pyg_batch, dist_masks, node_masks, merw_paths, merw_dists = _move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, _, aux_loss, _ = model(
                pyg_batch, dist_masks, node_masks,
                merw_paths=merw_paths, merw_dists=merw_dists,
            )
            task_loss = task.loss(logits, pyg_batch.y)
            loss = task_loss + aux_loss

        if is_train:
            loss.backward()
            # Track Edge-MERW gradient norm
            if getattr(model, "merw_path_encoder", None) is not None:
                tot_sq = 0.0
                for p in model.merw_path_encoder.parameters():
                    if p.grad is not None:
                        tot_sq += p.grad.detach().data.norm(2).item() ** 2
                model.merw_path_encoder._last_grad_norm = tot_sq ** 0.5

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(task_loss.item())
        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(pyg_batch.y))

    y_pred = np.concatenate(preds_acc, axis=0)
    y_true = np.concatenate(labels_acc, axis=0)
    metric = task.compute_metric(y_pred, y_true)
    return float(np.mean(losses)), float(metric)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
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
        use_merw=args.use_merw,
        merw_num_paths=args.merw_num_paths,
        merw_path_len=args.merw_path_len,
    )
    dataset_name = dataset_info["name"]

    task = build_task(dataset_name, dataset_info=dataset_info)
    task.loss_fn = task.loss_fn.to(args.device)

    print(f"[{dataset_name}] task={task.task_type} level={task.level} metric={task.metric_name}", flush=True)

    model = HopMaskedTransformerModelEdgeMERW(
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
        output_dim=task.output_dim,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=dataset_name,
        node_feat_dim=dataset_info.get("node_feat_dim"),
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        block_diag_out=args.block_diag_out,
        dynamic_cross_hop=args.dynamic_cross_hop,
        norm_type=args.norm_type,
        v_head_dim=args.v_head_dim,
        mask_type=args.mask_type,
        adj_self_loops=args.adj_self_loops,
        use_moe_gating=args.use_moe_gating,
        top_k=args.top_k,
        gate_noise=args.gate_noise,
        balance_coeff=args.balance_coeff,
        entropy_coeff=args.entropy_coeff,
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
        multihop_attn=args.multihop_attn,
        multihop_readout=args.multihop_readout,
        multihop_include_global=not args.multihop_no_global,
        embed_dropout=args.dropout,
        use_merw=args.use_merw,
        merw_num_paths=args.merw_num_paths,
        merw_path_len=args.merw_path_len,
        merw_merge=args.merw_merge,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.3f}M", flush=True)

    total_steps = max(args.max_epochs * len(train_loader), 1)
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=list(model.named_parameters()),
        lr_max=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    plateau_mode = "max" if task.higher_is_better else "min"
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=plateau_mode, factor=0.5, patience=args.reduce_lr_patience, min_lr=args.lr_min,
    )

    best_val = -float("inf") if task.higher_is_better else float("inf")
    best_test = None
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(args.max_epochs):
        t0 = time.time()
        tr_loss, tr_metric = run_epoch(model, train_loader, task, args.device, optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip)
        va_loss, va_metric = run_epoch(model, val_loader, task, args.device)
        te_loss, te_metric = run_epoch(model, test_loader, task, args.device)
        plateau_scheduler.step(va_metric)
        dt = time.time() - t0

        improved = (va_metric > best_val if task.higher_is_better else va_metric < best_val)
        if improved:
            best_val, best_test, best_epoch = va_metric, te_metric, epoch
            epochs_since_improve = 0
            ckpt_path = os.path.join(args.save_dir, f"best_{dataset_name}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val, "best_test": best_test}, ckpt_path)
        else:
            epochs_since_improve += 1

        ml = task.metric_label
        print(f"epoch {epoch:03d} | {dt:5.1f}s | "
              f"train loss {tr_loss:.4f} {ml} {tr_metric:.4f} | "
              f"val loss {va_loss:.4f} {ml} {va_metric:.4f} | "
              f"test loss {te_loss:.4f} {ml} {te_metric:.4f} | "
              f"best val {best_val:.4f} (epoch {best_epoch}, test {best_test})", flush=True)

        # Log Edge-MERW diagnostics
        merw_enc = getattr(model, "merw_path_encoder", None)
        if merw_enc is not None and getattr(merw_enc, "_diag", None) is not None:
            md = merw_enc._diag
            m_grad = getattr(merw_enc, "_last_grad_norm", 0.0)
            print(f"[epoch {epoch:03d}] Edge-MERW Diagnostics: "
                  f"attn_ent={md['entropy']:.3f} "
                  f"part={md['participation']:.2f}/{merw_enc.num_paths} "
                  f"top_path={md['top_path']:.3f} "
                  f"gate_alpha={md['gate']:.4f} "
                  f"grad_norm={m_grad:.4f}", flush=True)

        if epochs_since_improve >= args.patience:
            print(f"early stopping at epoch {epoch} (no val improvement in {args.patience} epochs)", flush=True)
            break

    print(f"BEST: val {ml} {best_val:.4f} | test {ml} {best_test:.4f} (epoch {best_epoch})", flush=True)


if __name__ == "__main__":
    main()
