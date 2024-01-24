#!/bin/bash 

oracle_array=('median2' 'osimertinib_mpo' \
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for seed in 0
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=3 python run.py reinvent_ls_gfn --config_default hparams_ls_gfn.yaml --task simple --oracle $oralce --wandb online --run_name re_iter --seed $seed
done
done
