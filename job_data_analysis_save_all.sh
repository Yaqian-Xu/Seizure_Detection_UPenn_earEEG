#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=200G
#SBATCH --time=120:00:00
#SBATCH --account=rrg-xilinliu
#SBATCH --mail-user=yaqian.xu@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=0412_balanced_save_all.out
#SBATCH --open-mode=append
module load python
module load scipy-stack
source "$HOME/projects/def-xilinliu/yqxu/ENV/bin/activate"
export PYTHONUNBUFFERED=1
python data_analysis_save_all.py
