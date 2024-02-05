# Genetic-guided GFlowNets

This repository provided implemented codes for the paper -- Genetic GFlowNets: Advancing in Practical Molecular Optimization Benchmark. 
> 

The codes are implemented our code based on the practical molecular optimization benchmark.
In addition, we implemented `Mol GA` and `GEGL` by adopting the original implementation; see the papers and codes below.

- [Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization (NeurIPS, 2022)](https://arxiv.org/abs/2206.12411)
(code: https://github.com/wenhao-gao/mol_opt)
- [Genetic algorithms are strong baselines for molecule generation](https://arxiv.org/abs/2310.09267)
(code: https://github.com/AustinT/mol_ga)
- [Guiding Deep Molecular Optimization with Genetic Exploration
 (NeurIPS, 2020)](https://proceedings.neurips.cc/paper/2020/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html)
(code: https://github.com/sungsoo-ahn/genetic-expert-guided-learning)



## Installation

Clone project and create environment with conda:
```
conda create -n genetic python==3.7
conda activate genetic

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c rdkit rdkit
pip install -r requirements.txt
```

**Note**: we highly recommend using Python 3.7, and PyTorch 1.12.1. Importantly, we use PyTDC 0.4.0 instead of 0.3.6.

For other baseline, please see `README_PMO.md`; we recommend using Python 3.8 for `GP BO` and `requirement_gpbo.txt`.


## Usage
```
CUDA_VISIBLE_DEVICES=0 python run.py genetic_gfn --task simple --wandb online --oracle qed --seed 0
```

To run genetic_GFN with SELFIES, use `genetic_gfn_selfies` instead of `genetic_gfn`.


