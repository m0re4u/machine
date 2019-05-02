#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH -t 20:00:00
#SBATCH --mem=12GB

#SBATCH --mail-type=END
#SBATCH --mail-user=michiel.vandermeer@student.uva.nl

#Load the modules needed for your environment
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176

# Go to machine directory
pushd ${HOME}/machine/

# Run training
python3 train_rl.py \
    --env-name BabyAI-PickupLoc-v0 \
    --print_every 1 \
    --segment_level word \
    --seed 1 \
    --slurm_id $SLURM_JOB_ID \
    --resume \
    --load_checkpoint models/BabyAI-GoToLocal-v0-_PPO_AC_expert_filmcnn_gru_mem_seed1_job2239329_19-05-01-13-58-20/008100_check.pt \
    --tb
