import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import Variable, seq_to_smiles, unique
from model import RNN
from data_structs import Vocabulary, Experience
import torch

from tdc import Evaluator
from polyleven import levenshtein

import itertools
import pickle
import pandas as pd


def diversity(smiles):
    # dist = [levenshtein(*pair) for pair in itertools.combinations(smiles, 2)]
    dist, normalized = [], []
    for pair in itertools.combinations(smiles, 2):
        dist.append(levenshtein(*pair))
        normalized.append(levenshtein(*pair)/max(len(pair[0]), len(pair[1])))
    evaluator = Evaluator(name = 'Diversity')
    mol_div = evaluator(smiles)
    return np.mean(normalized), np.mean(dist), mol_div


def novelty(new_smiles, ref_smiles):
    smiles_novelty = [min([levenshtein(d, od) for od in ref_smiles]) for d in new_smiles]
    smiles_norm_novelty = [min([levenshtein(d, od) / max(len(d), len(od)) for od in ref_smiles]) for d in new_smiles]
    evaluator = Evaluator(name = 'Novelty')
    mol_novelty = evaluator(new_smiles, ref_smiles)
    return np.mean(smiles_norm_novelty), np.mean(smiles_novelty), mol_novelty


class REINVENT_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent"

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

        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc)

        print("Model initialized, starting training...")

        step = 0
        patience = 0

        policy_smiles_norm_diversity, policy_smiles_diversity, policy_mol_diversity = [], [], []
        policy_smiles_norm_novelty, policy_smiles_novelty, policy_mol_novelty = [], [], []

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
            score = np.array(self.oracle(smiles))

            if step > 0:
                # import pdb; pdb.set_trace()
                smiles_norm_novelty, smiles_novelty, mol_novelty = novelty(smiles, experience.get_elems()[0])
                policy_smiles_norm_novelty.append(smiles_norm_novelty)
                policy_smiles_novelty.append(smiles_novelty)
                policy_mol_novelty.append(mol_novelty)
                smiles_norm_div, smiles_div, mol_div = diversity(smiles)
                policy_smiles_norm_diversity.append(smiles_norm_div)
                policy_smiles_diversity.append(smiles_div)
                policy_mol_diversity.append(mol_div)

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

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood.float() + config['sigma'] * Variable(score).float()
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            # First sample
            if config['experience_replay'] and len(experience)>config['experience_replay']:
                exp_seqs, exp_score, exp_prior_likelihood = experience.sample(config['experience_replay'])
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

            # Then add new experience
            prior_likelihood = prior_likelihood.data.cpu().numpy()
            new_experience = zip(smiles, score, prior_likelihood)
            experience.add_experience(new_experience)

            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Convert to numpy arrays so that we can print them
            augmented_likelihood = augmented_likelihood.data.cpu().numpy()
            agent_likelihood = agent_likelihood.data.cpu().numpy()

            step += 1

        results = {'exp_policy_smiles_novelty': policy_smiles_novelty, 
                    'exp_policy_smiles_norm_novelty': policy_smiles_norm_novelty, 
                    'exp_policy_mol_novelty': policy_mol_novelty, 
                    'policy_smiles_norm_diversity': policy_smiles_norm_diversity,
                    'policy_smiles_diversity': policy_smiles_diversity,
                    'policy_mol_diversity': policy_mol_diversity,}
        with open(f'./main/genetic_gfn/ga_results/reinvent_{oracle.name}_results_seed{self.seed}.pkl', 'wb') as f:
            pickle.dump(results, f)
