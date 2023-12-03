#!/bin/bash 

oracle_array=('drd2' 'qed' 'jnk3' 'gsk3b' 'celecoxib_rediscovery' 'troglitazone_rediscovery' \
        'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \
        'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo' \
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for seed in 0 1 2
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=0 python run.py gegl --task simple --oracle $oralce --wandb online --seed $seed
CUDA_VISIBLE_DEVICES=0 python run.py gegl --task simple --oracle $oralce --wandb online --seed $seed --run_name long --config_default 'hparams_default_longer.yaml'
CUDA_VISIBLE_DEVICES=0 python run.py gegl --task simple --oracle $oralce --wandb online --seed $seed --run_name tb --config_default 'hparams_default_tb.yaml'
CUDA_VISIBLE_DEVICES=0 python run.py gegl --task simple --oracle $oralce --wandb online --seed $seed --run_name tb_long --config_default 'hparams_default_tb_long.yaml'
CUDA_VISIBLE_DEVICES=0 python run.py genetic_gfn --task simple --oracle $oralce --wandb online --run_name canonical --seed $seed
CUDA_VISIBLE_DEVICES=0 python run.py genetic_gfn --task simple --oracle $oralce --wandb online --run_name canonical --seed $seed --config_default 'hparams_default_long.yaml'
done
done