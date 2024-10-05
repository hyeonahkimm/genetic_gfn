import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def get_acq_fn():

    return UCB



class AcquisitionFunctionWrapper():
    def __init__(self, model, l2r):
        self.model = model
        self.l2r = l2r

    def __call__(self, x):
        raise NotImplementedError()
    
    def update(self, data):
        self.fit(data)
    def fit(self, data):
        self.model.fit(data, reset=True)
    def save(self,path):
        torch.save(self.model.state_dict(),path)
    def load(self,path):
        self.model.load_state_dict(torch.load(path))


class NoAF(AcquisitionFunctionWrapper):
    def __call__(self, x):
        return self.l2r(self.model(x))
    def eval(self, x):
        mean, _ = self.model.eval(x)
        return self.l2r(mean)   
    
class UCB(AcquisitionFunctionWrapper):
    def __init__(self, model, l2r, args):
        super().__init__(model, l2r)
        self.kappa = args['kappa']
    
    def __call__(self, x):
        if self.kappa == 0.0:
            mean, _ = self.model.eval(x)
            return self.l2r(mean)
        mean, std = self.model.forward_with_uncertainty(x)
        return self.l2r(mean + self.kappa * std)