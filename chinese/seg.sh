#!/bin/bash
#SBATCH -J seg
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4

cd /home/cs_project_CTC_dual_fusion/datasets
source /home/.bashrc
conda activate pytorch_38

python lip_hand_seg.py
# python CCS_dataset.py