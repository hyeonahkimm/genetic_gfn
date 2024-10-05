import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from proxy.regression import Regressor, DropoutRegressor, EvidentialRegressor, EnsembleRegressor, GPRegressor
from mol_mdp_ext import MolMDPExtended
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.analytic import UpperConfidenceBound, ExpectedImprovement
# from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qExpectedImprovement
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.objective import GenericMCObjective
# from botorch.acquisition.objective import ScalarizedPosteriorTransform
# from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.sampling.samplers import SobolQMCNormalSampler
from sklearn.model_selection import train_test_split
from utils.acq_func import qUpperConfidenceBound, qExpectedImprovement
import time
from copy import copy, deepcopy


def make_proxy_model(args, mdp):
    repr_type = args.proxy_repr_type
    nemb = args.proxy_nemb
    num_conv_steps = args.proxy_num_conv_steps
    model_version = args.proxy_model_version

    if args.proxy_uncertainty == "none":
        model = Regressor(args,
                          nhid=nemb,
                          nvec=0,
                          num_out_per_stem=mdp.num_blocks,
                          num_out_per_mol=len(args.objectives),
                          num_conv_steps=num_conv_steps,
                          version=model_version,
                          dropout_rate=args.proxy_dropout)

    if args.proxy_uncertainty == "dropout":
        model = DropoutRegressor(args,
                                 nhid=nemb,
                                 nvec=0,
                                 num_out_per_stem=mdp.num_blocks,
                                 num_out_per_mol=len(args.objectives),
                                 num_conv_steps=num_conv_steps,
                                 version=model_version,
                                 dropout_rate=args.proxy_dropout,
                                 num_dropout_samples=args.proxy_num_dropout_samples)

    elif args.proxy_uncertainty == 'ensemble':
        model = EnsembleRegressor(args,
                                  nhid=nemb,
                                  nvec=0,
                                  num_out_per_stem=mdp.num_blocks,
                                  num_out_per_mol=len(args.objectives),
                                  num_conv_steps=num_conv_steps,
                                  version=model_version,
                                  dropout_rate=args.proxy_dropout,
                                  num_dropout_samples=args.proxy_num_dropout_samples)

    elif args.proxy_uncertainty == 'evidential':
        model = EvidentialRegressor(args,
                                    nhid=nemb,
                                    nvec=0,
                                    num_out_per_stem=mdp.num_blocks,
                                    num_out_per_mol=len(args.objectives),
                                    num_conv_steps=num_conv_steps,
                                    version=model_version,
                                    dropout_rate=args.proxy_dropout)
        
    elif args.proxy_uncertainty == 'GP':
        model = GPRegressor(args,
                            nhid=nemb,
                            nvec=0,
                            num_out_per_stem=mdp.num_blocks,
                            num_out_per_mol=len(args.objectives),
                            num_conv_steps=num_conv_steps,
                            version=model_version,
                            dropout_rate=args.proxy_dropout)
    
    model.to(args.device)
    if args.floatX == 'float64':
        model = model.double()

    return model

def get_proxy(args, bpath, oracle):
    if args.acq_fn.lower() == 'none':
        return NoAF(args, bpath, oracle)

    elif args.acq_fn.lower() == 'ucb':
        return UCB(args, bpath, oracle)
    
    elif args.acq_fn.lower() == 'ucb_chebyshev':
        return UCB_chebyshev(args, bpath, oracle)

    elif args.acq_fn.lower() == 'ei':
        return EI(args, bpath, oracle)

class Proxy:
    def __init__(self, args, bpath, oracle):
        self.args = args
        self.ref_point = torch.zeros(len(args.objectives)).to(args.device)
        self.oracle = oracle
        self.device = args.device
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(args.device, args.proxy_repr_type)
        if args.floatX == 'float64':
            self.mdp.floatX = torch.double
        else:
            self.mdp.floatX = torch.float
        self.init_model()

    def init_model(self):
        self.proxy = make_proxy_model(self.args, self.mdp)
        if self.args.proxy_uncertainty == 'ensemble':
            self.params = sum([list(model.parameters()) for model in self.proxy.proxy], [])
            self.opt = torch.optim.Adam(self.params, self.args.proxy_learning_rate,
                                        weight_decay=self.args.proxy_weight_decay)
        elif self.args.proxy_uncertainty == 'GP':
            pass
        else:
            self.opt = torch.optim.Adam(self.proxy.parameters(), self.args.proxy_learning_rate,
                                        weight_decay=self.args.proxy_weight_decay)

    def initialize_from_checkpoint(self):
        checkpoint = torch.load(
            self.args.proxy_init_checkpoint, map_location=self.device)
        self.proxy.proxy.load_state_dict(checkpoint)

        print('initialize from %s Done!' % self.args.proxy_init_checkpoint)

    def get_partitioning(self, dataset):
        ys = []
        for s, r in dataset.iterset(self.args.proxy_mbsize, 'train'):
            y = r
            ys.append(y)
        ys = torch.cat(ys, dim=0)        
        self.mean = torch.mean(ys, dim=0, keepdim=True)
        self.std = torch.std(ys, dim=0, keepdim=True)

        self.proxy.mean = self.mean
        self.proxy.std = self.std

        return FastNondominatedPartitioning(ref_point=self.ref_point, Y=ys)

    def update(self, dataset, round_idx, reset=False):
        print("Training surrogate function...")
        if reset:
            self.init_model()
        self.partitioning = self.get_partitioning(dataset)

        if self.args.proxy_uncertainty == 'GP':
            self.proxy.fit(dataset)
        else:
            self.proxy.fit(dataset, self.opt, self.mean, self.std, round_idx)

    def __call__(self, m, weights=None):
        raise NotImplementedError

class NoAF(Proxy):
    def __call__(self, m, weights=None):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        m.dtype = m.x.dtype

        objective = GenericMCObjective(get_chebyshev_scalarization(
            weights=weights.squeeze(), Y=torch.zeros(0, len(self.args.objectives))))
        mean = self.proxy.posterior(m).mean

        return ((weights * mean).sum(), mean.squeeze())

class UCB(Proxy):
    def __init__(self, args, bpath, oracle):
        super().__init__(args, bpath, oracle)
        self.beta = args.beta
        self.sampler = SobolQMCNormalSampler(128)
        self.score_clip = torch.tensor([0.6, 0.6, 0.7, 0.7]).unsqueeze(0).to(args.device)
        self.args = args
        
    def upper_confidence_bound(self, mu: np.array, var: np.array, beta: float): 
        return mu + (beta * var).sqrt()

    def __call__(self, m, weights=None):
        if self.args.proxy_uncertainty != 'GP':
            m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            m.dtype = m.x.dtype

        Y_bounds = torch.stack([self.partitioning.Y.min(
            dim=-2).values, self.partitioning.Y.max(dim=-2).values])
        posterior = self.proxy.posterior(m)
        mean = posterior.mean
        variance = posterior.variance   # oracle scale

        normalize_mean = normalize(mean, Y_bounds)   # [0, 1] scale
        new_mean = normalize_mean.matmul(weights.t()).squeeze()  # weighted_sum scalarization
        
        new_weights = weights / (Y_bounds[1]-Y_bounds[0])
        new_variance = (variance * new_weights**2).sum(1)
        
        raw_reward = self.upper_confidence_bound(mu=new_mean, var=new_variance, beta=self.beta)
        return raw_reward, mean.squeeze()

class UCB_chebyshev(Proxy):
    def __init__(self, args, bpath, oracle):
        super().__init__(args, bpath, oracle)
        self.beta = args.beta
        self.sampler = SobolQMCNormalSampler(128)

    def __call__(self, m, weights=None):
        if self.args.proxy_uncertainty != 'GP':
            m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            m.dtype = m.x.dtype

        Y_bounds = torch.stack([self.partitioning.Y.min(
            dim=-2).values, self.partitioning.Y.max(dim=-2).values])
        
        objective = GenericMCObjective(get_chebyshev_scalarization(
            weights=weights.squeeze(), Y=self.partitioning.Y))
        
        posterior = self.proxy.posterior(m)  # oracle scale
        mean = posterior.mean
        variance = posterior.variance

        # * chebyshev_scalarization
        acq_func = qUpperConfidenceBound(
            model=self.proxy,
            objective=objective,
            beta=self.beta,  # 0.1
            sampler=self.sampler)

        return (acq_func(m), mean.squeeze())

class EI(Proxy):
    def __init__(self, args, bpath, oracle):
        super().__init__(args, bpath, oracle)
        self.beta = args.beta
        self.sampler = SobolQMCNormalSampler(128)

    def __call__(self, m, weights=None):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        m.dtype = m.x.dtype

        Y_bounds = torch.stack([self.partitioning.Y.min(
            dim=-2).values, self.partitioning.Y.max(dim=-2).values])
        objective = GenericMCObjective(get_chebyshev_scalarization(
            weights=weights.squeeze(), Y=self.partitioning.Y))
        posterior = self.proxy.posterior(m)
        mean = posterior.mean
        variance = posterior.variance

        acq_func = qExpectedImprovement(
            model=self.proxy,
            objective=objective,
            best_f=torch.quantile(objective(self.partitioning.Y), 0.8),
            sampler=self.sampler
        )

        return (acq_func(m), mean.squeeze())
