import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import Variable, seq_to_smiles, unique
from model import RNN
from data_structs import Vocabulary, Experience, MolData
import torch
import wandb

from rdkit import Chem


class REINVENT_LS_GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent_ls_gfn"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        # optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        log_z = torch.nn.Parameter(torch.tensor([5.]).cuda())
        optimizer = torch.optim.Adam([{'params': Agent.rnn.parameters(), 
                                        'lr': config['learning_rate']},
                                    {'params': log_z, 
                                        'lr': config['lr_z']}])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc, max_size=config['num_keep'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0
            
            for _ in range(64 // (config['batch_size'] * (config['ls_iter']) + 1)):
                # Sample from Agent
                seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])

                # Remove duplicates, ie only consider unique seqs
                unique_idxs = unique(seqs)
                seqs = seqs[unique_idxs]
                agent_likelihood = agent_likelihood[unique_idxs]
                entropy = entropy[unique_idxs]

                # Get prior likelihood and score
                prior_likelihood, _ = Prior.likelihood(Variable(seqs))
                smiles = seq_to_smiles(seqs, voc)
                score = np.array(self.oracle(smiles))

                if self.finish:
                    print('max oracle hit')
                    break 

                prev_n_oracles = len(self.oracle)
                
                new_experience = zip(smiles, score)
                experience.add_experience(new_experience)

                # Local Search
                if config['canonicalize']:
                    encoded = []
                    for i, smi in enumerate(smiles):
                        try:
                            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)
                            tokenized = voc.tokenize(canonical)
                            encoded.append(Variable(voc.encode(tokenized)))
                        except:
                            encoded.append(seqs[i])
                    encoded = MolData.collate_fn(encoded)
                else:
                    encoded = seqs
                
                ls_avg_score = score.mean() / (config['ls_iter'] + 1)
                avg_accept_ratio = 0.
                # assert False
                for _ in range(config['ls_iter']):
                    if len(encoded) == 0: break
                    # print((encoded == 53).nonzero()[:, 1], encoded.shape)
                    # partial_len = int((encoded).nonzero()[:, 1].max()//2)
                    try:
                        partial_len = ((encoded == 53).nonzero()[:, 1].min(dim=0)[0]*0.5).long()
                    except:
                        partial_len = (encoded.shape[1]//2)
                    # print(encoded.shape, partial_len, (encoded == 0).nonzero()[:, 1].min(dim=0)[0])
                    destroyed_seqs = encoded[:, :partial_len].long()
                    # print('seq:', seqs[:2])
                    # print('encoded:', encoded[:2])
                    # print(destroyed_seqs[:2])
                    repaired_seqs, _, _ = Agent.sample_start_from(destroyed_seqs)
                    repaired_smiles = seq_to_smiles(repaired_seqs, voc)

                    if config['canonicalize']:
                        repaired_seqs = []
                        for i, smi in enumerate(smiles):
                            try:
                                canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                                tokenized = voc.tokenize(canonical)
                                repaired_seqs.append(Variable(voc.encode(tokenized)))
                            except:
                                repaired_seqs.append(seqs[i])
                        repaired_seqs = MolData.collate_fn(repaired_seqs)

                    repaired_score = np.array(self.oracle(repaired_smiles))
                    ls_avg_score += repaired_score.mean()/(config['ls_iter'] + 1)
                
                    try:
                        accept_mask = repaired_score > score  # size mismathes (rarely)
                    except:
                        accept_mask = torch.tensor([False] * len(repaired_score)).to(encoded.device)

                    accept_mask = torch.tensor(accept_mask).to(encoded.device)
                    avg_accept_ratio += accept_mask.sum() / config['ls_iter']

                    if repaired_seqs.shape[1] < encoded.shape[1]:
                        repaired_seqs = torch.cat([repaired_seqs, torch.zeros(encoded.shape[0], encoded.shape[1] - repaired_seqs.shape[1]).long().to(encoded.device)], dim=1)
                    else:
                        encoded = torch.cat([encoded, torch.zeros(encoded.shape[0], repaired_seqs.shape[1] - encoded.shape[1]).long().to(encoded.device)], dim=1)

                    # print(torch.tensor(accept_mask).sum())
                    encoded = torch.where(accept_mask[:, None], repaired_seqs, encoded)
                    # encoded_score = torch.where(torch.tensor(accept_mask)[:, None].to(encoded.device), repaired_seqs, encoded)

                    new_experience = zip(repaired_smiles, repaired_score)
                    experience.add_experience(new_experience)

                try:
                    wandb.log({'ls_avg_score': ls_avg_score, 
                            'sample_avg_score': score.mean(),
                            'accept_ratio':avg_accept_ratio})
                except:
                    pass
                # assert False

                if self.finish:
                    print('max oracle hit')
                    break
            
            if self.finish:
                break 
            
            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # early stopping
            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 100:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            # Experience Replay
            # First sample
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    if config['rank_coefficient'] >= 1:
                        # exp_seqs, exp_score = experience.quantile_uniform_sample(config['experience_replay'], 0.001)
                        exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], 0.01)
                    elif config['rank_coefficient'] > 0:
                        exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])
                        # exp_seqs, exp_score, exp_pb = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'], return_pb=~config['canonical'])
                    else:
                        exp_seqs, exp_score = experience.sample(config['experience_replay'])
                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                    reward = torch.tensor(exp_score).cuda()

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * config['beta']
                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # kl penalty
                    loss += config['kl_coefficient'] * (exp_agent_likelihood - prior_agent_likelihood).mean()

                    avg_loss += loss.item()/config['experience_loop']

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            step += 1