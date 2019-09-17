#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH -t 2:00:00
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
    --env-name BabyAI-TransferGoToObjMultiple0-v0 \
    --print_every 1 \
    --slurm_id $SLURM_JOB_ID \
    --tb \
    --seed 1 \
    --reasoning \
    --reason_coef 2 \
    --drop_diag \
    --diag_targets 21 \
    --load_checkpoint models/BabyAI-CustomGoToObjMultiple-v0-_PPO_IAC_expert_filmcnn_gru_mem_seed100_job2812456_19-08-07-22-50-02/021900_check.pt \
    --resume

