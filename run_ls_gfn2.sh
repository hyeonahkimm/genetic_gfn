#!/bin/bash 

oracle_array=(
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for seed in 4
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=3 python run.py reinvent_ls_gfn --config_default hparams_ls_gfn.yaml --task simple --oracle $oralce --wandb online --run_name iter --seed $seed
done
done
