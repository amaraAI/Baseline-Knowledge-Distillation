#!/bin/bash
#SBATCH --gres=gpu:p100:1#
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

#SBATCH --account=rrg-arbeltal
#SBATCH --nodes=1

#SBATCH --mail-user=anjun@cim.mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# --job-name=_
# --output=%x-%j.out
# --error=%x-%j.err

#rm *.err
#rm *.out
source ../environments/blkd/bin/activate

python3 -W ignore train_blkd.py \
--learning_rate 0.1 \
--batch_size 40 \
--name BLKD \
--model wideresnet \
--epochs 80 \
--seed 1 \
--T 10 \
--alphakd 0.7 \
--data_augmentation