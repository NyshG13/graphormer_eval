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

run_cora_opt_a() {
    echo "Starting Cora Option A (dim 64, 8 heads, 2 heads/hop) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_4hops_8heads.json \
        --max_hops 5 \
        --num_heads 8 \
        --hidden_dim 64 \
        --num_layers 2 \
        --ffn_ratio 2.0 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.6 \
        --weight_decay 0.001 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_opt_a_dim64_h8 \
        --seed 0 > log_cora_opt_a_dim64_h8.log 2>&1 &
    echo "Cora Option A launched in background (PID $!). Tail logs with: tail -f log_cora_opt_a_dim64_h8.log"
}

run_cora_opt_b() {
    echo "Starting Cora Option B (dim 128, 8 heads, head_dim 16) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_4hops_8heads.json \
        --max_hops 5 \
        --num_heads 8 \
        --hidden_dim 128 \
        --num_layers 2 \
        --ffn_ratio 2.0 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.6 \
        --weight_decay 0.001 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_opt_b_dim128_h8 \
        --seed 0 > log_cora_opt_b_dim128_h8.log 2>&1 &
    echo "Cora Option B launched in background (PID $!). Tail logs with: tail -f log_cora_opt_b_dim128_h8.log"
}

run_cora_opt_c() {
    echo "Starting Cora Option C (dim 128, 16 heads, 4 heads/hop) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_4hops_16heads.json \
        --max_hops 5 \
        --num_heads 16 \
        --hidden_dim 128 \
        --num_layers 2 \
        --ffn_ratio 2.0 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.6 \
        --weight_decay 0.001 \
        --lr 0.001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_opt_c_dim128_h16 \
        --seed 0 > log_cora_opt_c_dim128_h16.log 2>&1 &
    echo "Cora Option C launched in background (PID $!). Tail logs with: tail -f log_cora_opt_c_dim128_h16.log"
}

run_cora_exp4() {
    echo "Starting EXP 4: Cora Option C WITH Feature Normalization on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_4hops_16heads.json \
        --max_hops 5 \
        --num_heads 16 \
        --hidden_dim 128 \
        --num_layers 2 \
        --ffn_ratio 2.0 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.6 \
        --weight_decay 0.0005 \
        --lr 0.002 \
        --lr_min 0.0001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --normalize_features \
        --save_dir checkpoints_cora_exp4_norm_features \
        --seed 0 > log_cora_exp4_norm_features.log 2>&1 &
    echo "EXP 4 launched in background (PID $!). Tail logs with: tail -f log_cora_exp4_norm_features.log"
}

run_cora_exp5() {
    echo "Starting EXP 5: Cora Option C WITH Unequal Hop Allocation (8 on 1-hop, 4 on 2-hop, 2 on 3-hop, 2 on 4-hop) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_4hops_biased_16heads.json \
        --max_hops 5 \
        --num_heads 16 \
        --hidden_dim 128 \
        --num_layers 2 \
        --ffn_ratio 2.0 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.6 \
        --weight_decay 0.0005 \
        --lr 0.002 \
        --lr_min 0.0001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_exp5_unequal_hops \
        --seed 0 > log_cora_exp5_unequal_hops.log 2>&1 &
    echo "EXP 5 launched in background (PID $!). Tail logs with: tail -f log_cora_exp5_unequal_hops.log"
}

run_cora_exp6() {
    echo "Starting EXP 6: Cora Option C WITH Input Embed Dropout (0.5) on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_4hops_16heads.json \
        --max_hops 5 \
        --num_heads 16 \
        --hidden_dim 128 \
        --num_layers 2 \
        --ffn_ratio 2.0 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.6 \
        --embed_dropout 0.5 \
        --weight_decay 0.0005 \
        --lr 0.002 \
        --lr_min 0.0001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 25 \
        --no_random_global \
        --save_dir checkpoints_cora_exp6_embed_dropout \
        --seed 0 > log_cora_exp6_embed_dropout.log 2>&1 &
    echo "EXP 6 launched in background (PID $!). Tail logs with: tail -f log_cora_exp6_embed_dropout.log"
}

run_cora_combo_b() {
    echo "Starting Combination B: Cora Option C WITH Ego-Biased Hops (2 on hop 0, 8 on hop 1, 4 on hop 2, 2 on hop 3, NO hop 4) + Feature Norm on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train_hop_masked_light.py \
        --dataset cora \
        --hop_mode file \
        --hop_file cora_4hops_ego_biased_16heads.json \
        --max_hops 4 \
        --num_heads 16 \
        --hidden_dim 128 \
        --num_layers 2 \
        --ffn_ratio 2.0 \
        --num_parts 1 \
        --batch_size 1 \
        --dropout 0.6 \
        --weight_decay 0.0005 \
        --lr 0.002 \
        --lr_min 0.0001 \
        --max_epochs 500 \
        --patience 100 \
        --reduce_lr_patience 30 \
        --no_random_global \
        --normalize_features \
        --save_dir checkpoints_cora_combo_b_ego_biased \
        --seed 0 > log_cora_combo_b_ego_biased.log 2>&1 &
    echo "Combination B launched in background (PID $!). Tail logs with: tail -f log_cora_combo_b_ego_biased.log"
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
    cora_opt_a|opt_a)
        run_cora_opt_a
        ;;
    cora_opt_b|opt_b)
        run_cora_opt_b
        ;;
    cora_opt_c|opt_c)
        run_cora_opt_c
        ;;
    exp4|cora_exp4)
        run_cora_exp4
        ;;
    exp5|cora_exp5)
        run_cora_exp5
        ;;
    exp6|cora_exp6)
        run_cora_exp6
        ;;
    combo_b|cora_combo_b|b)
        run_cora_combo_b
        ;;
    all)
        run_exp13
        run_exp14
        run_exp15
        ;;
    *)
        echo "Usage: ./run_experiments_13_14_15.sh [13|13_no_aug|14|15|cora2|cora2_no_aug|cora3|cora3_no_aug|cora_opt_a|cora_opt_b|cora_opt_c|exp4|exp5|exp6|combo_b|all] [GPU_ID (default: 0)]"
        echo "Example: ./run_experiments_13_14_15.sh combo_b 0"
        exit 1
        ;;
esac
