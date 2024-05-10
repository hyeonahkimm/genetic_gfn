#!/bin/bash 


oracle_array=('troglitazone_rediscovery' \
        'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \
        'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' )

for seed in 0 1 2 3 4
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=1 python run.py genetic_gfn_al --task simple --wandb online --oracle $oralce --seed $seed --config_default 'hparams_al.yaml' --run_name noaf
done
done
