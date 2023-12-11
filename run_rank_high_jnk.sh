#!/bin/bash 

oracle_array=('jnk3')

for seed in 0 1 2
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=0 python run.py reinvent_ga --task simple --config_default 'hparams_rank3.yaml' --wandb online --run_name gen3_rank3_high_penalty --oracle $oralce --seed $seed
done
done
