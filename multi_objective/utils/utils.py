import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from botorch.utils.multi_objective.hypervolume import Hypervolume
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy import stats
from scipy.special import kl_div
import networkx as nx
import itertools
import time
import rdkit
from rdkit import Chem

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evidential_loss(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer
    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict
    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll  # torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg  # torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss

def tanimoto_distances(fp, other_fps):
    pairwise_sim = rdkit.DataStructs.BulkTanimotoSimilarity(fp, other_fps)
    return max(pairwise_sim)