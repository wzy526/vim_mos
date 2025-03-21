#!/bin/bash

source /cluster/home/wzy/anaconda3/etc/profile.d/conda.sh
conda activate mos
cd ~/vim_mos/vim;

MASTER_PORT=$(shuf -i 29500-65535 -n 1)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port $MASTER_PORT \
    --use_env main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 64 \
    --lr 1e-3 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --num_workers 25 \
    --data-path /home/data/imagenet/ \
    --output_dir ./output/mlp_mosvim_tiny_dstate64_#expert4_lr1e-3_decay5e-2_depth24_amp \
    --if_amp \
    --use_wandb True\
