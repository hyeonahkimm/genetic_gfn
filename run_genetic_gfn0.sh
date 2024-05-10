#!/bin/bash 

oracle_array=('jnk3' 'drd2' 'qed' 'gsk3b' 'celecoxib_rediscovery' )

for seed in 0 1 2 3 4
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=0 python run.py genetic_gfn_al --task simple --wandb online --oracle $oralce --seed $seed --config_default 'hparams_al.yaml' --run_name noaf
done
done
