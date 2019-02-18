#!/bin/bash
#SBATCH -N 2
#SBATCH -p gpu_short
#SBATCH -t 00:30:00

set -x

#Load the modules needed for your environment
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176 cuDNN/7.4.1.5-CUDA-9.0.176 NCCL/2.2.12-CUDA-9.0.176

pushd ${HOME}/machine/
python3 train_rl.py --env-name BabyAI-GoToLocal-v0 --print_every 1
