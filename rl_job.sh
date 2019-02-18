#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_short
#SBATCH -t 00:30:00

#Load the modules needed for your environment
module load Python/3.6.3-foss-2017b
module load CUDA/10.0.130

pushd ${HOME}/machine/
python3 train_rl.py --env-name BabyAI-GoToLocal-v0 --print_every 1
