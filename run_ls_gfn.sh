#!/bin/bash 

oracle_array=('drd2' 'qed' 'jnk3' 'gsk3b' 'celecoxib_rediscovery' 'troglitazone_rediscovery')

for seed in 0
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=3 python run.py reinvent_ls_gfn --config_default hparams_lsgfn_can.yaml --task simple --oracle $oralce --wandb online --run_name 4x4x4 --seed $seed
done
done
