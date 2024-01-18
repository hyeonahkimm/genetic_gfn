#!/bin/bash 

model="reinvent_ga"

nohup python -u run.py ${model} \
    --n_jobs 1 --task tune --n_runs 50 --wandb online --config_default hparams_default_lr.yaml --config_tune hparams_tune_512.yaml \
    --oracles jnk3 isomers_c9h10n2o2pf2cl zaleplon_mpo perindopril_mpo > tune_lr_${model}.out 2>&1 &

# CUDA_VISIBLE_DEVICES=1 bash tune.sh