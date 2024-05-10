#!/bin/bash 


oracle_array=('median1' 'median2' 'osimertinib_mpo' \
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' )

for seed in 0 1 2 3 4
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=6 python run.py genetic_gfn_al --task simple --wandb online --oracle $oralce --seed $seed --config_default 'hparams_al.yaml' --run_name noaf
done
done
