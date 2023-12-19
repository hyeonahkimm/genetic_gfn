#!/bin/bash 

oracle_array=('drd2' 'qed' 'jnk3' 'gsk3b' 'celecoxib_rediscovery' 'troglitazone_rediscovery' \
        'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \
        'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo' \
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for seed in 0 1 2 3 4
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=0 python run.py reinvent_ga --task simple --config_default 'hparams_tb.yaml' --wandb online --run_name rank2_prior_tb --oracle $oralce --seed $seed
done
done