#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=gpu
#SBATCH --time 20:20:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 10
#SBATCH --job-name="Adam_bs16_refined"
#SBATCH --output=%x.o%j

#< Charge resources to account   
#SBATCH --account t_2024_dlagm

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate ra2

python ./src/train.py --epochs=500 --bs=16 --early_stop=20 --lr=1e-4 --opt=Adam --refine_model=../models/Adam_lr1e-03_bs16_es20/best_model.pth

conda deactivate
