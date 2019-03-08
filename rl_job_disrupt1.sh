#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH -t 12:00:00

#Load the modules needed for your environment
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176

pushd ${HOME}/machine/
python3 train_rl.py \
    --env-name BabyAI-GoToLocal-v0 \
    --print_every 1 \
    --disrupt 1 \
    --slurm_id $SLURM_JOB_ID \
    --tb
