#!/bin/bash 

oracle_array=('drd2' 'qed')

for seed in 0 1 2
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=0 python run.py gflownet_al --task simple --oracle $oralce --wandb online --run_name gflownet_al --seed $seed
done
done
