#!/bin/bash 

oracle_array=('zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for seed in 0
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=3 python run.py reinvent_ls_gfn --config_default hparams_lsgfn_can.yaml --task simple --oracle $oralce --wandb online --run_name 4x4x4 --seed $seed
done
done
