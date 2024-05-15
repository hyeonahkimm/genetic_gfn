#!/bin/bash 


oracle_array=('scaffold_hop') 


for oralce in "${oracle_array[@]}"
do
for seed in 1 2 3 4
do
# echo $oralce
CUDA_VISIBLE_DEVICES=7 python run.py genetic_gfn_al --task simple --wandb online --oracle $oralce --seed $seed --config_default hparams_eps.yaml --run_name eps_noisy
done
done
