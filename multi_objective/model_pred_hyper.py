from tkinter import X
from typing import Callable, Tuple, Union, Dict

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

import torch_geometric.nn as gnn
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn.inits import reset, zeros
# from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from model_block import GraphAgent

class HyperPredictor(torch.nn.Module):
    def __init__(self, nemb, out_per_stem, out_per_mol, ray_hidden_dim=100, n_objectives=4):
        super().__init__()
        self.ray_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_objectives, ray_hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(ray_hidden_dim, ray_hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.stem_in_dim = nemb * 2
        self.stem2pred_dims = [nemb, nemb, out_per_stem]
        self.global_in_dim = nemb
        self.global2pred_dims = [nemb, out_per_mol]
        
        prvs_dim = self.stem_in_dim
        for i, dim in enumerate(self.stem2pred_dims):
            setattr(self, f"stem2pred_{i}_weights", torch.nn.Linear(ray_hidden_dim, prvs_dim * dim))
            setattr(self, f"stem2pred_{i}_bias", torch.nn.Linear(ray_hidden_dim, dim))
            prvs_dim = dim
        
        prvs_dim = self.global_in_dim
        for i, dim in enumerate(self.global2pred_dims):
            setattr(self, f"global2pred_{i}_weights", torch.nn.Linear(ray_hidden_dim, prvs_dim * dim))
            setattr(self, f"global2pred_{i}_bias", torch.nn.Linear(ray_hidden_dim, dim))
            prvs_dim = dim


    def forward(self, ray):  # ray: conditional weights
        stem2pred_out_dict = dict()
        global2pred_out_dict = dict()
        features = self.ray_mlp(ray)

        prvs_dim = self.stem_in_dim
        for i, dim in enumerate(self.stem2pred_dims):
            stem2pred_out_dict[f"fc_{i}_weights"] = self.__getattr__(f"stem2pred_{i}_weights")(
                features[0]
            ).reshape(dim, prvs_dim)
            stem2pred_out_dict[f"fc_{i}_bias"] = self.__getattr__(f"stem2pred_{i}_bias")(
                features[0]
            ).flatten()
            prvs_dim = dim
        
        prvs_dim = self.global_in_dim          
        for i, dim in enumerate(self.global2pred_dims):
            global2pred_out_dict[f"fc_{i}_weights"] = self.__getattr__(f"global2pred_{i}_weights")(
                features[0]
            ).reshape(dim, prvs_dim)
            global2pred_out_dict[f"fc_{i}_bias"] = self.__getattr__(f"global2pred_{i}_bias")(
                features[0]
            ).flatten()
            prvs_dim = dim

        return stem2pred_out_dict, global2pred_out_dict


class TargetStem2Pred(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x, stem2pred_weights):
        for i in range(int(len(stem2pred_weights) / 2)):
            x = F.linear(x, stem2pred_weights[f"fc_{i}_weights"], stem2pred_weights[f"fc_{i}_bias"])
            if i < int(len(stem2pred_weights) / 2) - 1:
                x = self.activation(x)
        return x


class TargetGlobal2Pred(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x, global2pred_weights):
        for i in range(int(len(global2pred_weights) / 2)):
            x = F.linear(x, global2pred_weights[f"fc_{i}_weights"], global2pred_weights[f"fc_{i}_bias"])
            if i < int(len(global2pred_weights) / 2) - 1:
                x = self.activation(x)
        return x
    
    
class TargetGraphAgent(GraphAgent):
    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, version='v4', partition_init=0, ray_hidden_dim=100,
                  n_objectives=4, logit_clipping=0.0):
        super().__init__(nemb, nvec, out_per_stem, out_per_mol,
                         num_conv_steps, mdp_cfg, version, partition_init)

        self.hnet = HyperPredictor(nemb, out_per_stem, out_per_mol, ray_hidden_dim=ray_hidden_dim, n_objectives=n_objectives)
        self.stem2pred = TargetStem2Pred()
        self.global2pred = TargetGlobal2Pred()
        self.logit_clipping = logit_clipping

    def forward(self, graph_data, vec_data=None, do_stems=True):
        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x)
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes)
        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] *
            graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2))
        out = graph_data.x
        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        elif self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out)
            stem2pred_weights, global2pred_weights = self.hnet(vec_data) 

        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(
                self.conv(out, graph_data.edge_index, graph_data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        if do_stems:
            stem_block_batch_idx = (
                torch.tensor(graph_data._slice_dict['x'], device=out.device)[
                    graph_data.stems_batch]
                + graph_data.stems[:, 0])
            if self.version == 'v1' or self.version == 'v4':
                stem_out_cat = torch.cat(
                    [out[stem_block_batch_idx], graph_data.stemtypes], 1)
            elif self.version == 'v2' or self.version == 'v3':
                stem_out_cat = torch.cat([out[stem_block_batch_idx],
                                          graph_data.stemtypes,
                                          vec_data[graph_data.stems_batch]], 1)

            stem_preds = self.stem2pred(stem_out_cat, stem2pred_weights)
        else:
            stem_preds = None

        mol_preds = self.global2pred(
            gnn.global_mean_pool(out, graph_data.batch), global2pred_weights)

        if self.logit_clipping > 0:
            stem_preds = self.logit_clipping * torch.tanh(stem_preds)
            mol_preds = self.logit_clipping * torch.tanh(mol_preds)
            
        return stem_preds, mol_preds