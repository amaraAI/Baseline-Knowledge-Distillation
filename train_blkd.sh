#!/bin/bash
#SBATCH --gres=gpu:v100:1#
#SBATCH --account=def-jjclark
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G     # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out


python3 -W ignore train_blkd.py \
--learning_rate 0.1 \
--batch_size 80 \
--name r18_bsz80_BLKD_seed_0_ \
--model resnet18 \
--epochs 200 \
--seed 0 \
--T 10 \
--alphakd 0.7 \
--data_augmentation 
