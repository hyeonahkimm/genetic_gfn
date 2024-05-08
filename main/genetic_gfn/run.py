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
from priority_queue import MaxRewardPriorityQueue
import torch
from rdkit import Chem
from tdc import Evaluator
from polyleven import levenshtein

import itertools
import pickle
import pandas as pd

from joblib import Parallel
from graph_ga_expert import GeneticOperatorHandler
# from smiles_ga_expert import GeneticOperatorHandler as SmilesGA


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


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


class Genetic_GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn"

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

        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        policy_smiles_norm_diversity, policy_smiles_diversity, policy_mol_diversity = [], [], []
        policy_smiles_norm_novelty, policy_smiles_novelty, policy_mol_novelty = [], [], []
        ga_smiles_norm_novelty, ga_smiles_novelty, ga_mol_novelty = [], [], []
        policy_ga_smiles_norm_novelty, policy_ga_smiles_novelty, policy_ga_mol_novelty = [], [], []
        parents_children_smiles_distances, parents_children_mol_distances = [], []
        best_scores = []

        tot_ga_results = pd.DataFrame({'children': [], 'parents': [], 'smiles_dist': [], 'mol_dist': []})

        prev_best = 0.
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
            # prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)
            if config['valid_only']:
                smiles = sanitize(smiles)
            
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

            # Calculate augmented likelihood
            # augmented_likelihood = prior_likelihood.float() + 500 * Variable(score).float()
            # reinvent_loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # print('REINVENT:', reinvent_loss.mean().item())

            # policy novelty
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
            policy_best = np.max(score)

            # Then add new experience
            new_experience = zip(smiles, score)
            experience.add_experience(new_experience)

            ga_best = 0.
            if config['population_size'] and len(self.oracle) > config['population_size']:
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))

                mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

                for g in range(config['ga_generations']):
                    child_smis, child_n_atoms, pop_smis, pop_scores, ga_results = ga_handler.query(
                            query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, 
                            rank_coefficient=config['rank_coefficient'], return_dist=True
                        )
                    
                    tot_ga_results = pd.concat([tot_ga_results, pd.DataFrame(ga_results)])

                    # policy novelty
                    smiles_norm_novelty, smiles_novelty, mol_novelty = novelty(child_smis, experience.get_elems()[0])
                    ga_smiles_norm_novelty.append(smiles_norm_novelty)
                    ga_smiles_novelty.append(smiles_novelty)
                    ga_mol_novelty.append(mol_novelty)
                    smiles_norm_novelty, smiles_novelty, mol_novelty = novelty(child_smis, smiles)
                    policy_ga_smiles_norm_novelty.append(smiles_norm_novelty)
                    policy_ga_smiles_novelty.append(smiles_novelty)
                    policy_ga_mol_novelty.append(mol_novelty)
                    # parents_children_smiles_distances.append(np.mean(ga_results['smiles_dist']))
                    # parents_children_mol_distances.append(np.mean(ga_results['mol_dist']))

                    child_score = np.array(self.oracle(child_smis))
                    if child_score.max() > ga_best:
                        ga_best = child_score.max()
                
                    new_experience = zip(child_smis, child_score)
                    experience.add_experience(new_experience)

                    mating_pool = (pop_smis+child_smis, pop_scores+child_score.tolist())

                    if self.finish:
                        print('max oracle hit')
                        break
            
            best_scores.append([policy_best, ga_best, prev_best])

            if max(ga_best, policy_best) > prev_best:
                prev_best = max(ga_best, policy_best)

            # Experience Replay
            # First sample
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    if config['rank_coefficient'] > 0:
                        exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])
                    else:
                        exp_seqs, exp_score = experience.sample(config['experience_replay'])

                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                    reward = torch.tensor(exp_score).cuda()

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * config['beta']
                    if  config['penalty'] == 'pb':
                        exp_backward_flow +=  prior_agent_likelihood

                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # KL penalty
                    if config['penalty'] == 'prior_kl':
                        loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                        loss += config['kl_coefficient']*loss_p

                    # print(loss.item())
                    avg_loss += loss.item()/config['experience_loop']

                    optimizer.zero_grad()
                    loss.backward()
                    # grad_norms = torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 1.0)
                    optimizer.step()

            step += 1
        
        best_scores = np.array(best_scores)

        results = {'exp_policy_smiles_novelty': policy_smiles_novelty, 
                    'exp_policy_smiles_norm_novelty': policy_smiles_norm_novelty, 
                    'exp_policy_mol_novelty': policy_mol_novelty, 
                    'exp_ga_smiles_norm_novelty': ga_smiles_norm_novelty, 
                    'exp_ga_smiles_novelty': ga_smiles_novelty, 
                    'exp_ga_mol_novelty': ga_mol_novelty,
                    'policy_ga_smiles_norm_novelty': policy_ga_smiles_norm_novelty, 
                    'policy_ga_smiles_novelty': policy_ga_smiles_novelty, 
                    'policy_ga_mol_novelty': policy_ga_mol_novelty,
                    'policy_smiles_norm_diversity': policy_smiles_norm_diversity,
                    'policy_smiles_diversity': policy_smiles_diversity,
                    'policy_mol_diversity': policy_mol_diversity,
                    'policy_best_scores': best_scores[:, 0],
                    'ga_best_scores': best_scores[:, 1],
                    'tot_best_scores': best_scores[:, 2],}
        
        if config['rank_coefficient'] < 0.1:
            with open(f'./main/genetic_gfn/ga_results/run_{oracle.name}_results_seed{self.seed}.pkl', 'wb') as f:
                pickle.dump(results, f)
            # results.to_pickle('./main/genetic_gfn/ga_results/run_' + oracle.name + '_results.pkl')
            tot_ga_results.to_pickle(f'./main/genetic_gfn/ga_results/run_{oracle.name}_ga_results_seed{self.seed}.pkl')
            tot_ga_results.to_csv(f'./main/genetic_gfn/ga_results/run_{oracle.name}_ga_results_seed{self.seed}.csv', index=False)

