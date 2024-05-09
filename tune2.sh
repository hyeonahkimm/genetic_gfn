#!/bin/bash 

model="genetic_gfn_al"

nohup python -u run.py ${model} \
    --n_jobs 1 --task tune --n_runs 50 --wandb online --config_default hparams_test.yaml --config_tune hparams_tune2.yaml \
    --oracles zaleplon_mpo perindopril_mpo > tune_proxy_${model}.out 2>&1 &

# CUDA_VISIBLE_DEVICES=1 bash tune.sh