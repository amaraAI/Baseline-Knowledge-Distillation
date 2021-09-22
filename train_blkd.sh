#!/bin/bash
#SBATCH --gres=gpu:p100:1#
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00

#SBATCH --account=rrg-arbeltal
#SBATCH --nodes=1

#SBATCH --mail-user=anjun@cim.mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#SBATCH --job-name=r18_bsz80_BLKD_seed_0_
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


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
