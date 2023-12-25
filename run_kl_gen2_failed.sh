#!/bin/bash 

oracle_array=('perindopril_mpo' 'amlodipine_mpo' 'sitagliptin_mpo')

for seed in 4
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=0 python run.py reinvent_ga --task simple --config_default 'hparams_kl_gen2.yaml' --wandb online --run_name rank2xgen2_kl3_rank2 --oracle $oralce --seed $seed
done
done
