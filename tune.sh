#!/bin/bash 

model="genetic_gfn_al"

python -u run.py ${model} \
    --n_jobs 1 --task tune --n_runs 50 --wandb online --config_default hparams_al.yaml --config_tune hparams_tune.yaml \
    --oracles zaleplon_mpo perindopril_mpo

# CUDA_VISIBLE_DEVICES=1 bash tune.sh