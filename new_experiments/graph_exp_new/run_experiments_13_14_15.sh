#!/usr/bin/env bash
# ==============================================================================
# Experiments 13, 14, 15: Extended Vanilla Hop-Masked Transformer Benchmarking
# ==============================================================================
#
# EXP 13: Vanilla hop-masked with tuned parameters WITHOUT path embeddings on Cora
#         - 1 intact graph partition (--num_parts 1, batch_size 1)
#         - 9 hops, 8 heads, 2 layers, hidden_dim 128, ffn_ratio 4.0
#         - dropout 0.5, weight_decay 5e-4, lr 1e-3
#         - max_epochs 500, patience 100, reduce_lr_patience 25
#         - Log: log_cora_vanilla_no_pemb.log
#
# EXP 14: Vanilla hop-masked with tuned parameters WITH path embeddings on Cora
#         - Same tuned architecture as EXP 13, but with --use_path_embeddings
#         - max_epochs 500, patience 100, reduce_lr_patience 25
#         - Log: log_cora_vanilla_pemb.log
#
# EXP 15: Vanilla hop-masked with tuned parameters on ZINC-12k
#         - hidden_dim 64, 4 layers, 4 heads, 1 global head (--num_global_heads 1)
#         - max_hops 10, batch_size 32, ffn_ratio 2.0, dropout 0.1
#         - Categorical bond encoder enabled (--use_edge_features)
#         - graph_pool mean, lr 1e-3, weight_decay 1e-5
#         - max_epochs 600, patience 100, reduce_lr_patience 25
#         - Log: log_zinc12k_edges_global_extended.log
# ==============================================================================

GPU=${2:-0}
CMD=$1

run_exp13() {
    echo "Starting EXP 13: Cora (Vanilla Tuned, No Path Embeddings) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hidden_dim 128 \
        --num_heads 8 \
        --num_layers 2 \
        --ffn_ratio 4.0 \
        --max_hops 9 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.5 \
        --weight_decay 0.0005 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --save_dir checkpoints_cora_vanilla_no_pemb \
        --seed 0 > log_cora_vanilla_no_pemb.log 2>&1 &
    echo "EXP 13 launched in background (PID $!). Tail logs with: tail -f log_cora_vanilla_no_pemb.log"
}

run_exp14() {
    echo "Starting EXP 14: Cora (Vanilla Tuned, With Path Embeddings) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hidden_dim 128 \
        --num_heads 8 \
        --num_layers 2 \
        --ffn_ratio 4.0 \
        --max_hops 9 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.5 \
        --weight_decay 0.0005 \
        --use_path_embeddings \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --save_dir checkpoints_cora_vanilla_pemb \
        --seed 0 > log_cora_vanilla_pemb.log 2>&1 &
    echo "EXP 14 launched in background (PID $!). Tail logs with: tail -f log_cora_vanilla_pemb.log"
}

run_exp15() {
    echo "Starting EXP 15: ZINC-12k (Vanilla Tuned with Edge Features & 1 Global Head) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset ZINC12k \
        --max_hops 10 \
        --hidden_dim 64 \
        --num_heads 4 \
        --num_global_heads 1 \
        --ffn_ratio 2.0 \
        --num_layers 4 \
        --dropout 0.1 \
        --use_edge_features \
        --graph_pool mean \
        --lr 0.001 \
        --weight_decay 1e-5 \
        --batch_size 32 \
        --max_epochs 600 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --save_dir checkpoints_zinc12k_edges_global_extended \
        --seed 0 > log_zinc12k_edges_global_extended.log 2>&1 &
    echo "EXP 15 launched in background (PID $!). Tail logs with: tail -f log_zinc12k_edges_global_extended.log"
}

run_cora_iter2() {
    echo "Starting Cora Iteration 2: Localized Walk Blending on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --blend_adj_power \
        --max_hops 4 \
        --num_heads 4 \
        --num_layers 2 \
        --ffn_ratio 4.0 \
        --num_parts 1 \
        --batch_size 1 \
        --hidden_dim 32 \
        --dropout 0.6 \
        --weight_decay 0.001 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --save_dir checkpoints_cora_iter2_walk_blend \
        --seed 0 > log_cora_iter2_walk_blend.log 2>&1 &
    echo "Cora Iteration 2 launched in background (PID $!). Tail logs with: tail -f log_cora_iter2_walk_blend.log"
}

run_cora_iter2_no_aug() {
    echo "Starting Cora Iteration 2 (WITHOUT Random Global Augmentation) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --blend_adj_power \
        --max_hops 4 \
        --num_heads 4 \
        --num_layers 2 \
        --ffn_ratio 4.0 \
        --num_parts 1 \
        --batch_size 1 \
        --hidden_dim 32 \
        --dropout 0.6 \
        --weight_decay 0.001 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_iter2_no_random_global \
        --seed 0 > log_cora_iter2_no_random_global.log 2>&1 &
    echo "Cora Iteration 2 (No Aug) launched in background (PID $!). Tail logs with: tail -f log_cora_iter2_no_random_global.log"
}

run_cora_iter3() {
    echo "Starting Cora Iteration 3: Compressed Non-Linear Stride on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_nonlinear_hops.json \
        --max_hops 9 \
        --num_heads 4 \
        --num_layers 2 \
        --ffn_ratio 4.0 \
        --num_parts 1 \
        --batch_size 1 \
        --hidden_dim 32 \
        --dropout 0.5 \
        --weight_decay 0.0005 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --save_dir checkpoints_cora_iter3_nonlinear_stride \
        --seed 0 > log_cora_iter3_nonlinear_stride.log 2>&1 &
    echo "Cora Iteration 3 launched in background (PID $!). Tail logs with: tail -f log_cora_iter3_nonlinear_stride.log"
}

run_cora_iter3_no_aug() {
    echo "Starting Cora Iteration 3: Compressed Non-Linear Stride (WITHOUT Random Global Augmentation) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_nonlinear_hops.json \
        --max_hops 9 \
        --num_heads 4 \
        --num_layers 2 \
        --ffn_ratio 4.0 \
        --num_parts 1 \
        --batch_size 1 \
        --hidden_dim 32 \
        --dropout 0.5 \
        --weight_decay 0.0005 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_iter3_no_random_global \
        --seed 0 > log_cora_iter3_no_random_global.log 2>&1 &
    echo "Cora Iteration 3 (No Aug) launched in background (PID $!). Tail logs with: tail -f log_cora_iter3_no_random_global.log"
}

run_exp13_no_aug() {
    echo "Starting EXP 13: Cora Vanilla Tuned (WITHOUT Random Global Augmentation) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hidden_dim 128 \
        --num_heads 8 \
        --num_layers 2 \
        --ffn_ratio 4.0 \
        --max_hops 9 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.5 \
        --weight_decay 0.0005 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_vanilla_no_pemb_no_random_global \
        --seed 0 > log_cora_vanilla_no_pemb_no_random_global.log 2>&1 &
    echo "EXP 13 (No Aug) launched in background (PID $!). Tail logs with: tail -f log_cora_vanilla_no_pemb_no_random_global.log"
}

case "$CMD" in
    13|exp13)
        run_exp13
        ;;
    13_no_aug|exp13_no_aug)
        run_exp13_no_aug
        ;;
    14|exp14)
        run_exp14
        ;;
    15|exp15)
        run_exp15
        ;;
    cora2|cora_iter2)
        run_cora_iter2
        ;;
    cora2_no_aug|cora_iter2_no_aug)
        run_cora_iter2_no_aug
        ;;
    cora3|cora_iter3)
        run_cora_iter3
        ;;
    cora3_no_aug|cora_iter3_no_aug)
        run_cora_iter3_no_aug
        ;;
    all)
        run_exp13
        run_exp14
        run_exp15
        ;;
    *)
        echo "Usage: ./run_experiments_13_14_15.sh [13|13_no_aug|14|15|cora2|cora2_no_aug|cora3|cora3_no_aug|all] [GPU_ID (default: 0)]"
        echo "Example: ./run_experiments_13_14_15.sh cora3_no_aug 0"
        exit 1
        ;;
esac
