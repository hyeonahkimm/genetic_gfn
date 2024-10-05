from logging import critical
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_pred_hyper
import model_block
import model_atom
import torch_geometric
from torch_scatter import scatter
from mol_mdp_ext import MolMDPExtended

def make_model(args, mdp, is_proxy=False):
    repr_type = args.proxy_repr_type if is_proxy else args.repr_type
    nemb = args.proxy_nemb if is_proxy else args.nemb
    num_conv_steps = args.proxy_num_conv_steps if is_proxy else args.num_conv_steps
    model_version = args.proxy_model_version if is_proxy else args.model_version
        
    if repr_type == 'block_graph':
        condition_type = args.condition_type
        if condition_type is None:
            model = model_block.GraphAgent(nemb=nemb,
                                        nvec=len(args.objectives),
                                        out_per_stem=mdp.num_blocks,
                                        out_per_mol=1,
                                        num_conv_steps=num_conv_steps,
                                        mdp_cfg=mdp,
                                        version='v4',
                                        partition_init=args.partition_init)

        elif condition_type == 'HN':
            model = model_pred_hyper.TargetGraphAgent(nemb=nemb,
                                            nvec=len(args.objectives),
                                            out_per_stem=mdp.num_blocks,
                                            out_per_mol=1,
                                            num_conv_steps=num_conv_steps,
                                            mdp_cfg=mdp,
                                            version='v4',
                                            partition_init=args.partition_init,
                                            ray_hidden_dim=args.ray_hidden_dim,
                                            n_objectives=args.n_objectives,
                                            logit_clipping=args.logit_clipping)
            
        elif condition_type == 'FiLM':
            model = model_block.GraphAgent_FiLM(nemb=nemb,
                                        nvec=len(args.objectives),
                                        out_per_stem=mdp.num_blocks,
                                        out_per_mol=1,
                                        num_conv_steps=num_conv_steps,
                                        mdp_cfg=mdp,
                                        version='v4',
                                        partition_init=args.partition_init)
            
        elif condition_type == 'concat':
            model = model_block.GraphAgent_Concat(nemb=nemb,
                                        nvec=len(args.objectives),
                                        out_per_stem=mdp.num_blocks,
                                        out_per_mol=1,
                                        num_conv_steps=num_conv_steps,
                                        mdp_cfg=mdp,
                                        version='v4',
                                        partition_init=args.partition_init)
    
    elif repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=num_conv_steps,
                                     version=model_version,
                                     do_nblocks=(hasattr(args,'include_nblocks')
                                                 and args.include_nblocks), dropout_rate=0.1)
    elif repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)

    model.to(args.device)
    if args.floatX == 'float64':
        model = model.double()

    return model

class FMGFlowNet(nn.Module):
    def __init__(self, args, bpath):
        super().__init__()
        self.args = args
        mdp = MolMDPExtended(bpath)
        mdp.post_init(args.device, args.repr_type,
                      include_nblocks=args.include_nblocks)
        mdp.build_translation_table()
        self.model = make_model(args, mdp, is_proxy=False)
        self.opt = torch.optim.Adam(self.model.parameters(
        ), args.learning_rate, weight_decay=args.weight_decay)

        self.loginf = 1000  # to prevent nans
        self.log_reg_c = args.log_reg_c
        self.balanced_loss = args.balanced_loss
        self.do_nblocks_reg = False
        self.max_blocks = args.max_blocks
        self.leaf_coef = args.leaf_coef
        self.clip_grad = args.clip_grad
        # self.score_criterion = nn.MSELoss(reduction='none')
        self.score_criterion = nn.MSELoss()

    def forward(self, graph_data, vec_data=None, do_stems=True):
        return self.model(graph_data, vec_data, do_stems)

    def train_step(self, p, pb, a, pw, w, r, s, d, mols, i):
        loss, term_loss, flow_loss = self.FMLoss(p, pb, a, pw, w, r, s, d)

        self.opt.zero_grad()
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad)
        self.opt.step()
        self.model.training_steps = i+1
        
        return (loss.item(), term_loss.item(), flow_loss.item())

    def FMLoss(self, p, pb, a, pw, w, r, s, d):
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]
        # state outputs
        stem_out_s, mol_out_s = self.model(s, w) # log(F)
        # parents of the state outputs
        stem_out_p, mol_out_p = self.model(p, pw)
        # index parents by their corresponding actions
        qsa_p = self.model.index_output_by_action(
            p, stem_out_p, mol_out_p[:, 0], a)
        # then sum the parents' contribution, this is the inflow
        exp_inflow = (torch.zeros((ntransitions,), device=qsa_p.device, dtype=qsa_p.dtype)
                      .index_add_(0, pb, torch.exp(qsa_p)))  # pb is the parents' batch index
        inflow = torch.log(exp_inflow + self.log_reg_c)
        # sum the state's Q(s,a), this is the outflow
        exp_outflow = self.model.sum_output(s, torch.exp(
            stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(self.log_reg_c + r + exp_outflow * (1-d))
        if self.do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) /
                                (s.nblocks * self.max_blocks)).pow(2)
        else:
            losses = _losses = (inflow - outflow_plus_r).pow(2)

        term_loss = (losses * d).sum() / (d.sum() + 1e-20)  # terminal nodes
        flow_loss = (losses * (1-d)).sum() / \
            ((1-d).sum() + 1e-20)  # non-terminal nodes
        
        if self.balanced_loss:
            loss = term_loss * self.leaf_coef + flow_loss
        else:
            loss = losses.mean()

        return loss, term_loss, flow_loss
    
class TBGFlowNet(nn.Module):
    def __init__(self, args, bpath):
        super().__init__()
        self.args = args
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(args.device, args.repr_type,
                           include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        self.model = make_model(args, self.mdp, is_proxy=False)
        self.Z = nn.Sequential(nn.Linear(len(args.objectives), args.nemb//2), nn.LeakyReLU(),
                                         nn.Linear(args.nemb//2, 1))
        self.Z.to(args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        self.opt_Z = torch.optim.Adam(self.Z.parameters(), args.Z_learning_rate, weight_decay=args.weight_decay)

    def forward(self, graph_data, vec_data=None, do_stems=True):
        return self.model(graph_data, vec_data, do_stems)

    def train_step(self, p, pb, a, pw, w, r, s, d, mols, i):
        loss = self.TBLoss(p, a, w, r, d, mols)
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        loss.backward()
        if self.args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.clip_grad)
        self.opt.step()
        self.opt_Z.step()

        return (loss.item(),)

    @property
    def Z(self):
        return self.model.Z

    def TBLoss(self, p, a, w, r, d, mols):
        # logit
        stem_out_p, mol_out_p = self.model(p, w)
        # index parents by their corresponding actions
        logits = -self.model.action_negloglikelihood(
            p, a, stem_out_p, mol_out_p)

        b = torch.cat([torch.tensor([0], device=logits.device),
                       torch.cumsum(d.long(), 0)[:-1]], dim=0)
        n = torch.tensor([len(self.mdp.parents(mol)) if a[idx, 0].item() != -1 else 1.
                                    for idx, mol in enumerate(mols[1])], device=logits.device)
        # n = torch.tensor([len(self.mdp.parents(mol)) for mol in mols[1]], device=logits.device)
        forward_ll = scatter(logits, b, reduce='sum')
        backward_ll = scatter(torch.log(1/n), b, reduce='sum')

        losses = ((self.Z(w[d==1.]) + forward_ll) - (torch.log(r[d == 1.]) + backward_ll)).pow(2) 
        loss = losses.mean()

        return loss

class MOReinforce(TBGFlowNet):
    def TBLoss(self, p, a, w, r, d, mols):
        # logit
        stem_out_p, mol_out_p = self.model(p, w)
        # index parents by their corresponding actions
        logits = -self.model.action_negloglikelihood(
            p, a, stem_out_p, mol_out_p)

        b = torch.cat([torch.tensor([0], device=logits.device),
                       torch.cumsum(d.long(), 0)[:-1]], dim=0)
        n = torch.tensor([len(self.mdp.parents(mol)) if a[idx, 0].item() != -1 else 1.
                                    for idx, mol in enumerate(mols[1])], device=logits.device)
        # n = torch.tensor([len(self.mdp.parents(mol)) for mol in mols[1]], device=logits.device)
        forward_ll = scatter(logits, b, reduce='sum')

        rewards = r[d == 1.]
        losses = forward_ll * (-rewards - (-1) * rewards.mean())
        loss = losses.mean()

        return loss
    


