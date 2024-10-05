# Sample-efficient Multi-objective Molecular Optimization with GFlowNets
Official implementation of NeurIPS'23 paper "[Sample-efficient Multi-objective Molecular Optimization with GFlowNets](https://arxiv.org/abs/2302.04040)". This code is built on top of the [GFlowNet repo](https://github.com/GFNOrg/gflownet/tree/master).

## Environment:

- torch 
- numpy 
- scipy 
- tqdm
- pymoo  
- botorch 
- gpytorch
- pandas
- rdkit 
- torch-geometric
- h5py 
- ray
- scikit-learn                                 
- tensorboard                      

## Experiments:

### Single-round synthetic scenario

```
python main.py --condition_type HN --enable_tensorboard
```

### Multi-objective Bayesian Optimization

```
python main_mobo.py --objectives gsk3b,jnk3,qed,sa --alpha_vector 3,3,1,1 --save --enable_tensorboard
```

## Citation
If you find this repository useful, please consider citing our work:
```
@inproceedings{
  zhu2023sampleefficient,
  title={Sample-efficient Multi-objective Molecular Optimization with {GF}lowNets},
  author={Yiheng Zhu and Jialu Wu and Chaowen Hu and Jiahuan Yan and Chang-Yu Hsieh and Tingjun Hou and Jian Wu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=uoG1fLIK2s}
}
```

