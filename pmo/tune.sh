#!/bin/bash 

model="genetic_gfn"

python -u run.py ${model} \
    --n_jobs 1 --task tune --n_runs 50 --wandb online --config_default hparams_pb.yaml --config_tune hparams_tune.yaml \
    --oracles zaleplon_mpo perindopril_mpo

# CUDA_VISIBLE_DEVICES=1 bash tune.sh