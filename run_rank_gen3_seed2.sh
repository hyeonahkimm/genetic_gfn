#!/bin/bash 

oracle_array=('drd2' 'qed' 'jnk3' 'gsk3b' 'celecoxib_rediscovery' 'troglitazone_rediscovery' \
        'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \
        'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo' \
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for seed in 2
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=5 python run.py reinvent_ga --task simple --config_default 'hparams_blended_ga.yaml' --wandb online --run_name gen3_blended --oracle $oralce --seed $seed
done
done
