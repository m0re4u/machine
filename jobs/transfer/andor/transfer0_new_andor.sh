#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH -t 1:00:00
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
    --env-name BabyAI-TransferGoToObjAndOr0-v0 \
    --print_every 1 \
    --slurm_id $SLURM_JOB_ID \
    --tb \
    --seed 1 \
    --reasoning \
    --reason_coef 2 \
    --diag_targets 21 \
    --load_checkpoint models/BabyAI-CustomGoToObjAndOr-v0-_PPO_IAC_expert_filmcnn_gru_mem_seed1_job2604666_19-06-26-22-00-59/005900_check.pt \
    --resume

