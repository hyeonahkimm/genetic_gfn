import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, seq_to_selfies 
from model import RNN
from data_structs import Vocabulary, Experience
import torch

import selfies as sf
from joblib import Parallel
from graph_ga_expert import GeneticOperatorHandler

from rdkit import Chem
from tdc.chem_utils import MolConvert
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')
smiles2selfies = MolConvert(src = 'SMILES', dst = 'SELFIES')


class Genetic_GFN_SELFIES_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn_selfies"

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
            ##### original 
            # smiles = seq_to_smiles(seqs, voc) #################### matrix (seq) -> smiles_list
            # score = np.array(self.oracle(smiles))
            ##### original 

            ##### new 
            selfies_list = seq_to_selfies(seqs, voc) 
            smiles_list = selfies2smiles(selfies_list)
            # can_selfies, can_smiles = canonicalize(selfies_list)
            # score = np.array(self.oracle(can_smiles))
            score = np.array(self.oracle(smiles_list))
            ##### new 

            # print('NN:', score.max(), score.mean(), score.std(), len(score))

            if self.finish:
                print('max oracle hit')
                break 

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience*2:
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

            new_experience = zip(selfies_list, score) ## new 
            experience.add_experience(new_experience)

            if config['population_size'] and len(self.oracle) > config['population_size']:
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                # print(list(self.oracle.mol_buffer))
                mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

                for g in range(config['ga_generations']):
                    child_selfies, child_smiles, pop_smis, pop_scores = ga_handler.query(
                            query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, 
                            rank_coefficient=config['rank_coefficient'],
                        )

                    # child_smis = list(set(child_smis))
                    child_score = np.array(self.oracle(child_smiles))
      
                    new_experience = zip(child_selfies, child_score)
                    experience.add_experience(new_experience)

                    mating_pool = (pop_smis+child_smiles, pop_scores+child_score.tolist())
                    # print('GA' + str(g+1) + ':', child_score.max(), child_score.mean(), child_score.std(), len(child_smiles))

                    if self.finish:
                        print('max oracle hit')
                        break
                
            # TB
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])

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

