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


def canonicalize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            canonicalized.append(s)
    return canonicalized


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
            if config['canonicalize']:
                smiles = canonicalize(smiles)
            score = np.array(self.oracle(smiles))

            if self.finish:
                print('max oracle hit')
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
                if stuck_cnt >= 10:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)
            
            new_experience = zip(smiles, score)
            experience.add_experience(new_experience)

            # Local Search
            if config['canonicalize']:
                encoded = []
                for i, smile in enumerate(smiles):
                    try:
                        tokenized = voc.tokenize(smile)
                        encoded.append(Variable(voc.encode(tokenized)))
                    except:
                        pass
                encoded = MolData.collate_fn(encoded)
            else:
                encoded = seqs
            # min_len = torch.nonzero(encoded)[:, 1].min()
            partial_len = int(torch.nonzero(encoded)[:, 1].min()//2)
            destroyed_seqs = encoded[:, :partial_len]
            repaired_seqs, _, _ = Agent.sample_start_from(destroyed_seqs)
            repaired_smiles = seq_to_smiles(repaired_seqs, voc)
            if config['canonicalize']:
                repaired_smiles = canonicalize(repaired_smiles)
            repaired_score = np.array(self.oracle(repaired_smiles))
            accept_mask = repaired_score > score
            # print(len(repaired_smiles), accept_mask.sum())

            new_experience = zip([repaired_score[j] for j in accept_mask.nonzero()], repaired_score[accept_mask])
            experience.add_experience(new_experience)

            # assert False

            if self.finish:
                print('max oracle hit')
                break
            
            
            # Experience Replay
            # First sample
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    if config['rank_coefficient'] >= 1:
                        # exp_seqs, exp_score = experience.quantile_uniform_sample(config['experience_replay'], 0.001)
                        exp_seqs, exp_score, _ = experience.rank_based_sample(config['experience_replay'], 0.001)
                    elif config['rank_coefficient'] > 0:
                        exp_seqs, exp_score, exp_pb = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'], return_pb=~config['canonicalize'])
                        # exp_seqs, exp_score, exp_pb = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'], return_pb=~config['canonical'])
                    else:
                        exp_seqs, exp_score = experience.sample(config['experience_replay'])
                    # exp_seqs, exp_smis, exp_score = experience.sample_batch(config['experience_replay']) 
                    # tokenized = [voc.tokenize(smile) for smile in smiles]
                    # encoded = [Variable(voc.encode(tokenized_i)) for tokenized_i in tokenized]
                    # encoded = MolData.collate_fn(encoded)
                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())
                    # exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                    # exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                    # loss = torch.cat((loss, exp_loss), 0)
                    # agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

                    reward = torch.tensor(exp_score).cuda()
                    if config['penalty'] == 'kl':
                        # print((exp_agent_likelihood - prior_agent_likelihood).mean())
                        reward -= 0.001 * (exp_agent_likelihood - prior_agent_likelihood)

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * config['beta']
                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # Add regularizer that penalizes high likelihood for the entire sequence (from REINVENT)
                    if config['penalty'] == 'REINVENT':
                        # print(torch.nn.functional.mse_loss(exp_agent_likelihood, prior_agent_likelihood))
                        loss_p = - (1 / exp_agent_likelihood).mean()
                        # print('penalty:', loss_p.item())
                        loss += 1e3 * loss_p
                    elif config['penalty'] == 'prior_l2':
                        loss_p = torch.nn.functional.mse_loss(exp_agent_likelihood, prior_agent_likelihood)
                        loss += 1e-2 * loss_p
                    elif config['penalty'] == 'prior_kl':
                        loss_p = (exp_agent_likelihood.exp() * (exp_agent_likelihood - prior_agent_likelihood)).sum()
                        # print(loss.item())
                        # print(loss_p.item(), 1e-2 * torch.nn.functional.mse_loss(exp_agent_likelihood, prior_agent_likelihood).item(), 1e3 * (1 / exp_agent_likelihood).mean().item())
                        loss += 1e2 * loss_p
                    elif config['penalty'] == 'full_kl':
                        loss_p = torch.nn.functional.kl_div(exp_agent_likelihood, prior_agent_likelihood, log_target=True, reduction="none").sum(-1)
                        # print(loss_p.mean().item())
                        loss += loss_p.mean()

                    # print(loss.item())
                    avg_loss += loss.item()/config['experience_loop']

                    optimizer.zero_grad()
                    loss.backward()
                    # grad_norms = torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 1.0)
                    optimizer.step()
            # print(avg_loss)

            # Calculate loss
            # loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            # loss_p = - (1 / agent_likelihood).mean()
            # loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # Convert to numpy arrays so that we can print them
            # augmented_likelihood = augmented_likelihood.data.cpu().numpy()
            # agent_likelihood = agent_likelihood.data.cpu().numpy()

            step += 1

