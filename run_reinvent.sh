#!/bin/bash 

oracle_array=('jnk3' 'qed' 'gsk3b')

for seed in 1 2 3 4 0
do
for oracle in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=6 python run.py reinvent --task simple --oracle $oracle --seed $seed
done
done