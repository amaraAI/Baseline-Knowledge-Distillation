#!/bin/bash
#SBATCH --gres=gpu:v100:1#
#SBATCH --account=def-##### write the account you are using
#SBATCH --cpus-per-task=4 
#SBATCH --mem=64G    
#SBATCH --time=4:00:00
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
