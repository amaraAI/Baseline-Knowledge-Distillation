# BASELINE KNOWLEDGE DISTILLATION
This projects performs the Baseline Knowledge Distillation (BLKD) from the paper "Distilling the Knowledge in a Neural Network" by Hinton et al on CIFAR100 dataset. 

# RUNNING THE CODE
**1- Under the same folder create the following folder names:**
*/blkd_checkpoints*
*/blkd_checkpoints/best*
*/blkd_checkpoints_seeds*
*/blkd_tb*
*/jason_logs_blkd*

**2-Download the pretrained teacher network:**
The teacher network is a pre-trained Resnet152 on CIFAR100. The weights can be downloaded here: --link

**3-run:**
please follow the slurm file attached to this project: *"train_blkd.sh"*

