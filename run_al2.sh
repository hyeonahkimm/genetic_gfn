#!/bin/bash 

oracle_array=('median1' 'median2' 'osimertinib_mpo' \
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for seed in 0 
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=6 python run.py genetic_gfn_al --task simple --wandb online --oracle $oralce --seed $seed --config_default hparams_test.yaml --run_name kappa0
done
done
