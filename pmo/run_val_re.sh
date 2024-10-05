#!/bin/bash 


for seed in 0 1 2 3 4
do
# echo $oralce
CUDA_VISIBLE_DEVICES=1 python run.py gegl --task simple --seed $seed --oracle valsartan_smarts
done
