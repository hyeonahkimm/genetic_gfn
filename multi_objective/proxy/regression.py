from copy import copy, deepcopy
from statistics import mean
from utils.chem import atomic_numbers
from utils import chem
from utils.utils import evidential_loss
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, Set2Set
import torch.nn.functional as F
import torch.nn as nn
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
# from gpytorch.lazy.block_diag_lazy_tensor import BlockDiagLazyTensor
from gpytorch.lazy import lazify
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.model_list_gp_regression import ModelListGP
# from botorch.acquisition.multi_objective.objective import (
#     AnalyticMultiOutputObjective,
#     IdentityAnalyticMultiOutputObjective,
# )
from proxy.fingerprints import smiles_to_fp_array
from proxy.tanimoto_gp import TanimotoGP
from proxy.gp_utils import fit_gp_hyperparameters
import functools
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

import numpy as np
import pandas as pd
import subprocess
import gzip
import pickle
import pdb
import threading
import os.path as osp
import os
import time
import sys
import warnings
warnings.filterwarnings('ignore')


class MPNNet_v2(nn.Module):
    def __init__(self, num_feat=14, num_vec=3, dim=64,
                 num_out_per_mol=1, num_out_per_stem=105,
                 num_out_per_bond=1,
                 num_conv_steps=12, version='v1', dropout_rate=None):
        super().__init__()
        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_ops = num_out_per_stem
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps
        self.version = int(version[1:])
        self.dropout = nn.Dropout(dropout_rate)
        print('v:', self.version)
        assert 1 <= self.version <= 6

        if self.version < 5:
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.SiLU()

        if self.version < 4:
            net = nn.Sequential(nn.Linear(4, 128), self.act,
                                nn.Linear(128, dim * dim))
            self.conv = NNConv(dim, dim, net, aggr='mean')
        elif self.version == 4 or self.version == 6:
            self.conv = gnn.TransformerConv(dim, dim, edge_dim=4)
        else:
            self.convs = nn.Sequential(*[gnn.TransformerConv(dim, dim, edge_dim=4)
                                         for i in range(num_conv_steps)])

        # if self.version >= 6:
        #    self.g_conv = gnn.TransformerConv(dim, dim, heads=4)

        if self.version < 3:
            self.gru = nn.GRU(dim, dim)

        if self.version < 4:
            self.lin1 = nn.Linear(dim, dim * 8)
            self.lin2 = nn.Linear(dim * 8, num_out_per_stem)
        else:
            self.stem2out = nn.Sequential(nn.Linear(dim * 2, dim), self.act,
                                          nn.Linear(dim, dim), self.act,
                                          nn.Linear(dim, num_out_per_stem))
            #self.stem2out = nn.Sequential(nn.Linear(dim * 2, num_out_per_stem))

        if self.version < 3:
            self.set2set = Set2Set(dim, processing_steps=3)
        if self.version < 4:
            self.lin3 = nn.Linear(dim * 2 if self.version <
                                  3 else dim, num_out_per_mol)
        else:
            self.lin3 = nn.Sequential(nn.Linear(dim, dim), self.act,
                                      nn.Linear(dim, dim), self.act,
                                      nn.Linear(dim, num_out_per_mol))
        self.bond2out = nn.Sequential(nn.Linear(dim * 2, dim), self.act,
                                      nn.Linear(dim, dim), self.act,
                                      nn.Linear(dim, num_out_per_bond))

    def forward(self, data, vec_data=None, do_stems=True, do_bonds=False, k=None):
        if self.version == 1:
            batch_vec = vec_data[data.batch]
            out = self.act(self.lin0(torch.cat([data.x, batch_vec], 1)))
        elif self.version > 1:
            out = self.act(self.lin0(data.x))
        h = out.unsqueeze(0)
        h = self.dropout(h)

        if self.version < 4:
            for i in range(self.num_conv_steps):
                m = self.act(self.conv(out, data.edge_index, data.edge_attr))
                m = self.dropout(m)
                out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
                h = self.dropout(h)
                out = out.squeeze(0)
        elif self.version == 4 or self.version == 6:
            for i in range(self.num_conv_steps):
                out = self.act(self.conv(out, data.edge_index, data.edge_attr))
        else:
            for i in range(self.num_conv_steps):
                out = self.act(self.convs[i](
                    out, data.edge_index, data.edge_attr))
        if self.version >= 4:
            global_out = gnn.global_mean_pool(out, data.batch)

        if do_stems:
            # Index of the origin atom of each stem in the batch, we
            # need to adjust for the batch packing)
            stem_batch_idx = (
                torch.tensor(data.__slices__['x'], device=out.device)[
                    data.stems_batch]
                + data.stems)
            stem_atom_out = out[stem_batch_idx]
            # if self.version >= 6:
            #    per_stem_out = self.g_conv(stem)
            #    import pdb; pdb.set_trace()
            if self.version >= 4:
                stem_atom_out = torch.cat(
                    [stem_atom_out, global_out[data.stems_batch]], 1)
                per_stem_out = self.stem2out(stem_atom_out)
            else:
                per_stem_out = self.lin2(self.act(self.lin1(stem_atom_out)))
        else:
            per_stem_out = None

        if do_bonds:
            bond_data = out[data.bonds.flatten()].reshape(
                (data.bonds.shape[0], -1))
            per_bond_out = self.bond2out(bond_data)

        if self.version < 3:
            global_out = self.set2set(out, data.batch)
            global_out = self.dropout(global_out)
        per_mol_out = self.lin3(global_out)  # per mol scalar outputs

        if hasattr(data, 'nblocks'):
            per_stem_out = per_stem_out * \
                data.nblocks[data.stems_batch].unsqueeze(1)
            per_mol_out = per_mol_out * data.nblocks.unsqueeze(1)
            if do_bonds:
                per_bond_out = per_bond_out * data.nblocks[data.bonds_batch]

        if do_bonds:
            return per_stem_out, per_mol_out, per_bond_out
        return per_stem_out, per_mol_out


class MyPosterior:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


class Regressor(nn.Module):
    def __init__(self, args, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version, dropout_rate=0, do_stem_mask=True, do_nblocks=False):
        nn.Module.__init__(self)
        self.args = args
        self.training_steps = 0
        # atomfeats + stem_mask + atom one hot + nblocks
        num_feat = (14 + int(do_stem_mask) +
                    len(atomic_numbers) + int(do_nblocks))

        self.proxy = MPNNet_v2(
            num_feat=num_feat,
            num_vec=nvec,
            dim=nhid,
            num_out_per_mol=num_out_per_mol,
            num_out_per_stem=num_out_per_stem,
            num_conv_steps=num_conv_steps,
            version=version,
            dropout_rate=dropout_rate)

    def fit(self, dataset, opt):
        last_losses = []
        train_losses = []
        test_losses = []
        time_start = time.time()
        time_last_check = time.time()
        best_test_loss = 1000
        mbsize = self.args.proxy_mbsize
        early_stop_tol = self.args.proxy_early_stop_tol
        early_stop_count = 0

        self.proxy.train()
        for i in range(self.args.proxy_num_iterations+1):
            s, r = dataset.sample2batch(dataset.sample(mbsize))
            # s.x s.edge_index, s.edge_attr, s.stems: stem_atmidxs

            stem_out_s, mol_out_s = self.proxy(s, None, do_stems=False)
            loss = F.mse_loss(mol_out_s, r)
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.proxy.training_steps = i + 1

            if not i % 50:
                train_loss = [np.round(np.mean(i), 4)
                              for i in zip(*last_losses)]
                last_losses = []

                total_test_loss = 0

                self.proxy.eval()
                for s, r in dataset.iterset(max(mbsize, 64), mode='test'):
                    with torch.no_grad():
                        stem_o, mol_o = self.proxy(s, None, do_stems=False)
                        loss = F.mse_loss(mol_o, r, reduction='sum')
                        total_test_loss += loss.item()
                self.proxy.train()

                test_loss = total_test_loss / \
                    (len(dataset.test_mols)*len(self.args.objectives))
                test_losses.append(test_loss)
                print('Iter {}: Train Loss {}, Test Loss {}, Time {}'.format(
                    i, train_loss[0], round(test_loss, 4), round(time.time() - time_last_check, 3)))
                time_last_check = time.time()

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model = deepcopy(self.proxy)
                    best_model.to('cpu')
                    early_stop_count = 0
                    if self.args.save:
                        self.save(self.args.log_dir)

                else:
                    early_stop_count += 1
                    print('Early stop count: {}'.format(early_stop_count))

                if early_stop_count >= early_stop_tol:
                    print('Early stopping! Training time: {}, Best test loss: {}'.format(
                        time.time()-time_start, best_test_loss))
                    break

        self.proxy = deepcopy(best_model)
        self.proxy.to(self.args.device)

    def forward(self, graph, vec=None, do_stems=True, do_bonds=False, k=None):
        return self.proxy(graph, vec, do_stems=do_stems, do_bonds=do_bonds, k=k)

    def posterior(self, x):
        self.proxy.eval()
        with torch.no_grad():
            outputs = self.forward(x, None, do_stems=False)[1].squeeze(0)

        posterior = MyPosterior(outputs, torch.zeros_like(outputs))

        return posterior

    def save(self, checkpoint_dir):
        checkpoint_path = os.path.join(
            checkpoint_dir, f"proxy_init_checkpoint.pth")
        torch.save(self.proxy.state_dict(), checkpoint_path)


class DropoutRegressor(Regressor):
    def __init__(self, args, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version, \
        dropout_rate=0, num_dropout_samples=25, do_stem_mask=True, do_nblocks=False):
        super().__init__(args, nhid, nvec, num_out_per_stem, num_out_per_mol,
                         num_conv_steps, version, dropout_rate, do_stem_mask, do_nblocks)
        self.proxy_num_dropout_samples = num_dropout_samples

    def posterior(self, x):
        self.proxy.train()
        with torch.no_grad():
            outputs = torch.cat([self.forward(x, None, do_stems=False)[1].unsqueeze(0)
                                 for _ in range(self.proxy_num_dropout_samples)])

        posterior = MyPosterior(outputs.mean(dim=0), outputs.var(dim=0))

        return posterior


class EvidentialRegressor(nn.Module):
    def __init__(self, args, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version, dropout_rate=0, do_stem_mask=True, do_nblocks=False):
        nn.Module.__init__(self)
        self.args = args
        self.training_steps = 0
        # atomfeats + stem_mask + atom one hot + nblocks
        num_feat = (14 + int(do_stem_mask) +
                    len(atomic_numbers) + int(do_nblocks))

        self.proxy = MPNNet_v2(
            num_feat=num_feat,
            num_vec=nvec,
            dim=nhid,
            num_out_per_mol=num_out_per_mol*4,
            num_out_per_stem=num_out_per_stem,
            num_conv_steps=num_conv_steps,
            version=version,
            dropout_rate=dropout_rate)

    def fit(self, dataset, opt, mean, std, round_idx):
        self.mean = mean
        self.std = std
        
        last_losses = []
        train_losses = []
        test_losses = []
        time_start = time.time()
        time_last_check = time.time()
        best_test_loss = 1000
        mbsize = self.args.proxy_mbsize
        early_stop_tol = self.args.proxy_early_stop_tol
        early_stop_count = 0
        
        stop_event = threading.Event()
        sampler = dataset.start_samplers(1, mbsize)

        def stop_everything():
            stop_event.set()
            print('joining')
            dataset.stop_samplers_and_join()

        self.proxy.train()
        for i in range(self.args.proxy_num_iterations+1):
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_event.set()
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
            s, r = r
            r = (r - mean) / std
            # s.x s.edge_index, s.edge_attr, s.stems: stem_atmidxs
            
            # if bounds is not None:
            #     r = normalize(r, bounds)
            means, lambdas, alphas, betas = self.forward(s, None, do_stems=False)
            # the larger the lam, the larger the variance
            loss = evidential_loss(means, lambdas, alphas, betas, r, lam=self.args.evidential_lam).mean()
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.proxy.training_steps = i + 1

            if not i % 50:
                train_loss = [np.round(np.mean(i), 4)
                              for i in zip(*last_losses)]
                last_losses = []

                total_test_loss = 0
                total_normalize_test_loss = 0

                self.proxy.eval()
                for s, r in dataset.iterset(max(mbsize, 64), mode='test'):
                    with torch.no_grad():
                        means, lambdas, alphas, betas = self.forward(s, None, do_stems=False)
                        # if bounds is not None:
                        #     means = unnormalize(means, bounds)
                        normalize_loss = F.mse_loss(means, (r-mean)/std, reduction='sum')
                        total_normalize_test_loss += normalize_loss.item()
                        means = means * std + mean
                        loss = F.mse_loss(means, r, reduction='sum')
                        total_test_loss += loss.item()
                self.proxy.train()

                test_loss = total_test_loss / \
                    (len(dataset.test_mols)*len(self.args.objectives))
                normalize_test_loss = total_normalize_test_loss / \
                    (len(dataset.test_mols)*len(self.args.objectives))
                test_losses.append(test_loss)
                print('Iter {}: Train Loss {}, Test Loss {}, Normalize Test Loss {}, Time {}'.format(
                    i, train_loss[0], round(test_loss, 4), round(normalize_test_loss, 4), round(time.time() - time_last_check, 3)))
                time_last_check = time.time()

                if normalize_test_loss < best_test_loss:
                    best_test_loss = normalize_test_loss
                    best_model = deepcopy(self.proxy)
                    best_model.to('cpu')
                    early_stop_count = 0
                    if self.args.save:
                        self.save(self.args.log_dir, round_idx)

                else:
                    early_stop_count += 1
                    print('Early stop count: {}'.format(early_stop_count))

                if early_stop_count >= early_stop_tol:
                    print('Early stopping! Training time: {}, Best test loss: {}'.format(
                        time.time()-time_start, best_test_loss))
                    break
                
        stop_everything()
        self.proxy = deepcopy(best_model)
        self.proxy.to(self.args.device)

    def forward(self, graph, vec=None, do_stems=True, do_bonds=False, k=None):
        _, mol_out_s = self.proxy(graph, vec, do_stems=do_stems,
                            do_bonds=do_bonds, k=k)
        min_val = 1e-6
        means, loglambdas, logalphas, logbetas = torch.split(
            mol_out_s, mol_out_s.shape[1]//4, dim=1)
        lambdas = F.softplus(loglambdas) + min_val
        alphas = F.softplus(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = F.softplus(logbetas) + min_val

        return means, lambdas, alphas, betas
        
    def posterior(self, X, posterior_transform=None):
        self.proxy.eval()
        with torch.no_grad():
            means, lambdas, alphas, betas = self.forward(X, None, do_stems=False)
            inverse_evidence = 1. / ((alphas-1) * lambdas)
            vars = betas * inverse_evidence
        
        means = means * self.std + self.mean
        vars = vars * self.std ** 2
        
        # vars = BlockDiagLazyTensor(torch.diag(vars.squeeze()).unsqueeze(0))
        covariance_matrix = lazify(torch.diag(vars.squeeze()))
        mvn = MultitaskMultivariateNormal(means, covariance_matrix)
        
        posterior = GPyTorchPosterior(mvn)

        if posterior_transform is not None:
            return posterior_transform(posterior) 
        return posterior

    def save(self, checkpoint_dir, round_idx):
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{round_idx}_proxy_checkpoint.pth")
        torch.save(self.proxy.state_dict(), checkpoint_path)


class EnsembleRegressor(nn.Module):
    def __init__(self, args, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version, \
        dropout_rate=0, num_dropout_samples=5, do_stem_mask=True, do_nblocks=False):
        nn.Module.__init__(self)
        self.training_steps = 0
        # atomfeats + stem_mask + atom one hot + nblocks
        num_feat = (14 + int(do_stem_mask) +
                    len(atomic_numbers) + int(do_nblocks))
        self.proxy_num_dropout_samples = num_dropout_samples
        self.args = args
        self.device = args.device
        self.proxy = [MPNNet_v2(
                        num_feat=num_feat,
                        num_vec=nvec,
                        dim=nhid,
                        num_out_per_mol=num_out_per_mol,
                        num_out_per_stem=num_out_per_stem,
                        num_conv_steps=num_conv_steps,
                        version=version,
                        dropout_rate=dropout_rate).to(self.device) \
                            for _ in range(self.proxy_num_dropout_samples)]
        
    def fit(self, dataset, opt, mean, std, round_idx):
        self.mean = mean
        self.std = std
        
        last_losses = []
        train_losses = []
        test_losses = []
        time_start = time.time()
        time_last_check = time.time()
        best_test_loss = 1000
        mbsize = self.args.proxy_mbsize
        early_stop_tol = self.args.proxy_early_stop_tol
        early_stop_count = 0
        
        for i in range(self.args.proxy_num_iterations+1):
            s, r = dataset.sample2batch(dataset.sample(mbsize))
            r = (r - mean) / std  # (batch_size, num_obj)
            mol_out_s = self._call_models_train(s).mean(0)
            
            loss = F.mse_loss(mol_out_s, r)
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.training_steps = i + 1
            
            if not i % 50:
                train_loss = [np.round(np.mean(i), 4)
                              for i in zip(*last_losses)]
                last_losses = []

                total_test_loss = 0

                for s, r in dataset.iterset(max(mbsize, 64), mode='test'):
                    with torch.no_grad():
                        mol_o = self._call_models_eval(s).mean(0)
                        loss = F.mse_loss(mol_o, (r-mean)/std, reduction='sum')
                        total_test_loss += loss.item()

                test_loss = total_test_loss / \
                    (len(dataset.test_mols)*len(self.args.objectives))
                test_losses.append(test_loss)
                print('Iter {}: Train Loss {}, Test Loss {}, Time {}'.format(
                    i, train_loss[0], round(test_loss, 4), round(time.time() - time_last_check, 3)))
                time_last_check = time.time()

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    # best_model = deepcopy(self.proxy)
                    # best_model.to('cpu')
                    best_params = [[i.data.cpu().numpy() for i in model.parameters()] for model in self.proxy]
                    early_stop_count = 0
                    if self.args.save:
                        self.save(self.args.log_dir, round_idx)

                else:
                    early_stop_count += 1
                    print('Early stop count: {}'.format(early_stop_count))

                if early_stop_count >= early_stop_tol:
                    print('Early stopping! Training time: {}, Best test loss: {}'.format(
                        time.time()-time_start, best_test_loss))
                    break
        
        # load best parameters     
        for i, model in enumerate(self.proxy):
            for i, besti in zip(model.parameters(), best_params[i]):
                i.data = torch.tensor(besti).to(self.device)
        # self.args.logger.save(self.args.save_path, self.args)
    
    def _call_models_train(self, x):
        for model in self.proxy:
            model.train()
        ys = torch.stack([model(x, None, do_stems=False)[1] for model in self.proxy], dim=0)  # (5, 64, 2)
        return ys
    
    def _call_models_eval(self, x):
        for model in self.proxy:
            model.eval()
        ys = torch.stack([model(x, None, do_stems=False)[1] for model in self.proxy], dim=0)
        return ys
    
    def posterior(self, x):
        with torch.no_grad():
            outputs = self._call_models_eval(x)
        posterior = MyPosterior(outputs.mean(dim=0), outputs.var(dim=0))
        posterior.mean = posterior.mean * self.std + self.mean
        posterior.variance = posterior.variance * self.std ** 2
        return posterior
    
    def save(self, checkpoint_dir, round_idx):
        for i, model in enumerate(self.proxy):
            checkpoint_path = os.path.join(checkpoint_dir, f"{round_idx}_proxy_checkpoint_model_{i}.pth")
            torch.save(model.state_dict(), checkpoint_path)   


class GPRegressor(nn.Module):
    def __init__(self, args, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version, \
        dropout_rate=0, num_dropout_samples=5, do_stem_mask=True, do_nblocks=False):
        nn.Module.__init__(self)
        self.training_steps = 0
        # atomfeats + stem_mask + atom one hot + nblocks
        num_feat = (14 + int(do_stem_mask) +
                    len(atomic_numbers) + int(do_nblocks))
        self.proxy_num_dropout_samples = num_dropout_samples
        self.args = args
        self.device = args.device
        # self.objective = AnalyticMultiOutputObjective()
        self.NP_DTYPE = np.float32
        fingerprint_func = functools.partial(
            rdMolDescriptors.GetMorganFingerprintAsBitVect,
            radius=self.args.fp_radius,
            nBits=self.args.fp_nbits,
        )
        self.my_smiles_to_fp_array = functools.partial(
            smiles_to_fp_array, fingerprint_func=fingerprint_func
        )
        
    def fit(self, dataset):
        x_train = np.stack([self.my_smiles_to_fp_array(s) for s in dataset.smis]).astype(self.NP_DTYPE)  # (200, 1024)
        y_train = pd.DataFrame.from_dict(dataset.scores).values.astype(self.NP_DTYPE)  # (200, num_obj)
        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        self.proxy = self.get_trained_gp(X_train=x_train, y_train=y_train).to(self.device)
    
    def get_trained_gp(self, X_train, y_train):
        models = []
        for i in range(y_train.shape[-1]):
            obj = y_train[:, i]
            models.append(TanimotoGP(train_x=X_train, train_y=obj))  # input should be tensor
        model = ModelListGP(*models)
        
        fit_gp_hyperparameters(model)
        
        return model
    
    def posterior(self, x): 
        x = self.my_smiles_to_fp_array(Chem.MolToSmiles(x.mol))
        x = torch.as_tensor(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            posterior = self.proxy.posterior(x)  #! oracle scale
        return posterior
    

def mol2graph(mol, mdp, floatX=torch.float, bonds=False, nblocks=False):
    rdmol = mol.mol
    if rdmol is None:
        g = Data(x=torch.zeros((1, 14 + len(atomic_numbers))),
                 edge_attr=torch.zeros((0, 4)),
                 edge_index=torch.zeros((0, 2)).long())
    else:
        atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol.mol, ifcoord=False,
                                                    one_hot_atom=True, donor_features=False)
        g = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
    stems = mol.stem_atmidxs
    if not len(stems):
        stems = [0]
    stem_mask = torch.zeros((g.x.shape[0], 1))
    stem_mask[torch.tensor(stems).long()] = 1
    g.stems = torch.tensor(stems).long()
    if nblocks:
        nblocks = (torch.ones((g.x.shape[0], 1,)).to(floatX) *
                   ((1 + mdp._cue_max_blocks - len(mol.blockidxs)) / mdp._cue_max_blocks))
        g.x = torch.cat([g.x, stem_mask, nblocks], 1).to(floatX)
        g.nblocks = nblocks[0] * mdp._cue_max_blocks
    else:
        g.x = torch.cat([g.x, stem_mask], 1).to(floatX)
    g.edge_attr = g.edge_attr.to(floatX)
    if bonds:
        if len(mol.jbonds):
            g.bonds = torch.tensor(mol.jbond_atmidxs).long()
        else:
            g.bonds = torch.zeros((1, 2)).long()
    if g.edge_index.shape[0] == 0:
        g.edge_index = torch.zeros((2, 1)).long()
        g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).to(floatX)
        g.stems = torch.zeros((1,)).long()
    return g


def mols2batch(mols, mdp):
    batch = Batch.from_data_list(
        mols, follow_batch=['stems', 'bonds'])
    batch.to(mdp.device)
    return batch

# class MolAC_GCN(nn.Module):
#     def __init__(self, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version, dropout_rate=0, do_stem_mask=True, do_nblocks=False):
#         nn.Module.__init__(self)
#         self.training_steps = 0
#         # atomfeats + stem_mask + atom one hot + nblocks
#         num_feat = (14 + int(do_stem_mask) + len(atomic_numbers) + int(do_nblocks))
#         self.mpnn = MPNNet_v2(
#             num_feat=num_feat,
#             num_vec=nvec,
#             dim=nhid,
#             num_out_per_mol=num_out_per_mol,
#             num_out_per_stem=num_out_per_stem,
#             num_conv_steps=num_conv_steps,
#             version=version,
#             dropout_rate=dropout_rate)

#     def out_to_policy(self, s, stem_o, mol_o):
#         stem_e = torch.exp(stem_o)
#         mol_e = torch.exp(mol_o[:, 0])
#         Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
#         return mol_e / Z, stem_e / Z[s.stems_batch, None]

#     def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
#         stem_e = torch.exp(stem_o)
#         mol_e = torch.exp(mol_o[:, 0])
#         Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e
#         mol_lsm = torch.log(mol_e / Z)
#         stem_lsm = torch.log(stem_e / Z[s.stems_batch, None])
#         stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_lsm.device)
#         return -(
#             stem_lsm[stem_slices + a[:, 1]][
#                 torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
#             + mol_lsm * (a[:, 0] == -1))

#     def index_output_by_action(self, s, stem_o, mol_o, a):
#         stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
#         return (
#             stem_o[stem_slices + a[:, 1]][
#                 torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
#             + mol_o * (a[:, 0] == -1))
#     #(stem_o[stem_slices + a[:, 1]][torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0) + mol_o * (a[:, 0] == -1))

#     def sum_output(self, s, stem_o, mol_o):
#         return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o

#     def forward(self, graph, vec=None, do_stems=True, do_bonds=False, k=None, do_dropout=False):
#         return self.mpnn(graph, vec, do_stems=do_stems, do_bonds=do_bonds, k=k, do_dropout=do_dropout)

#     def _save(self, checkpoint_dir):
#         checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
#         torch.save(self.model.state_dict(), checkpoint_path)
#         return checkpoint_path

#     def _restore(self, checkpoint_path):
#         self.model.load_state_dict(torch.load(checkpoint_path))

# def mol2graph(mol, mdp, floatX=torch.float, bonds=False, nblocks=False):
#     rdmol = mol.mol
#     if rdmol is None:
#         g = Data(x=torch.zeros((1, 14 + len(atomic_numbers))),
#                  edge_attr=torch.zeros((0, 4)),
#                  edge_index=torch.zeros((0, 2)).long())
#     else:
#         atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol.mol, ifcoord=False,
#                                                     one_hot_atom=True, donor_features=False)
#         g = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
#     stems = mol.stem_atmidxs
#     if not len(stems):
#         stems = [0]
#     stem_mask = torch.zeros((g.x.shape[0], 1))
#     stem_mask[torch.tensor(stems).long()] = 1
#     g.stems = torch.tensor(stems).long()
#     if nblocks:
#         nblocks = (torch.ones((g.x.shape[0], 1,)).to(floatX) *
#                    ((1 + mdp._cue_max_blocks - len(mol.blockidxs)) / mdp._cue_max_blocks))
#         g.x = torch.cat([g.x, stem_mask, nblocks], 1).to(floatX)
#         g.nblocks = nblocks[0] * mdp._cue_max_blocks
#     else:
#         g.x = torch.cat([g.x, stem_mask], 1).to(floatX)
#     g.edge_attr = g.edge_attr.to(floatX)
#     if bonds:
#         if len(mol.jbonds):
#             g.bonds = torch.tensor(mol.jbond_atmidxs).long()
#         else:
#             g.bonds = torch.zeros((1,2)).long()
#     if g.edge_index.shape[0] == 0:
#         g.edge_index = torch.zeros((2, 1)).long()
#         g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).to(floatX)
#         g.stems = torch.zeros((1,)).long()
#     return g


# def mols2batch(mols, mdp):
#     batch = Batch.from_data_list(
#         mols, follow_batch=['stems', 'bonds'])
#     batch.to(mdp.device)
#     return batch
