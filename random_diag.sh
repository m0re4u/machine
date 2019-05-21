set -eu


END=10
for i in $(seq 1 $END);
do
python3 eval_rl.py \
  --env BabyAI-CustomGoToObjSmall-v0 \
  --model models/BabyAI-CustomGoToObjSmall-v0-_PPO_AC_expert_filmcnn_gru_mem_seed1_job2300034_19-05-14-18-08-30/007600_check.pt \
  --vocab models/BabyAI-CustomGoToObjSmall-v0-_PPO_AC_expert_filmcnn_gru_mem_seed1_job2300034_19-05-14-18-08-30/vocab.json  \
  --reasoning diagnostic \
  --seed $i \
  --episodes 25;
done
