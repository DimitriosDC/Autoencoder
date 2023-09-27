#!/bin/bash
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

module load Anaconda3/5.3.0

module load cuDNN/7.6.4.38-gcccuda-2019b

source activate autoencoder

cd autoencoder_project

python test_program.py $SLURM_ARRAY_TASK_ID