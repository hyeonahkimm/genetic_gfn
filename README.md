# Genetic-guided GFlowNets

This repository provided implemented codes for the paper -- Genetic GFlowNets: Advancing in Practical Molecular Optimization Benchmark. 
> 

The codes are implemented our code based on the practical molecular optimization benchmark.
In addition, we implemented `Mol GA` and `GEGL` by adopting the original implementation; see the papers and codes below.

- Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization ([paper](https://arxiv.org/abs/2206.12411), [code](https://github.com/wenhao-gao/mol_opt))
- Genetic algorithms are strong baselines for molecule generation ([paper](https://arxiv.org/abs/2310.09267), [code](https://github.com/AustinT/mol_ga))
- Guiding Deep Molecular Optimization with Genetic Exploration ([paper](https://proceedings.neurips.cc/paper/2020/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html), [code](https://github.com/sungsoo-ahn/genetic-expert-guided-learning))

Also, our codes are based on the following papers.
- Multi-objecetive: Sample-efficient Multi-objective Molecular Optimization with GFlowNets ([paper](https://arxiv.org/abs/2302.04040), [code](https://github.com/violet-sto/HN-GFN))
- SARS-CoV-2: De novo Drug Design using Reinforcement Learning with Multiple GPT Agent ([paper](https://arxiv.org/abs/2401.06155), [code](https://github.com/HXYfighter/MolRL-MGPT))


## Installation (PMO)

Clone project and create environment with conda:
```
conda create -n genetic python==3.7
conda activate genetic

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c rdkit rdkit
pip install -r requirements.txt
```

**Note**: we highly recommend using Python 3.7, and PyTorch 1.12.1. Importantly, we use PyTDC 0.4.0 instead of 0.3.6.

For other baseline, please see `README_PMO.md`;
When you use Python 3.8, you can install rdkit with `pip install rdkit-pypi`.

To run other experiments, including multi-objective and SARS-CoV-2, please see `README.md` in each directory and their original repository.


## Usage
#### PMO benchmark

```
cd pmo
python run.py genetic_gfn --task simple --wandb online --oracle qed --seed 0
```

To run genetic_GFN with SELFIES, use `genetic_gfn_selfies` instead of `genetic_gfn`.


#### Multi-objective

```
cd multi_objective
python run.py genetic_gfn --alpha_vector 3,4,2,1 --seed 0
python run.py genetic_gfn --alpha_vector 1,1 --objectives gsk3b,jnk3 --seed 0
```


#### SARS-CoV-2

```
cd sars_cov2
python genetic_gfn/train.py genetic_gfn --oracle docking_RdRp_mpo --wandb online --rank_coefficient 0.05 --seed 0
python genetic_gfn/train.py genetic_gfn --oracle docking_PLPro_7JIR_mpo --wandb online --rank_coefficient 0.05 --seed 0
```


