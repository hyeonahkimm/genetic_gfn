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
from tqdm import tqdm
from time import perf_counter

from joblib import Parallel
from graph_ga_expert import GeneticOperatorHandler

from proxy.proxy import get_proxy_model
from proxy.acquisition_fn import get_acq_fn


# proxy contruction code following GFN-AL
def construct_proxy(tokenizer,num_token,max_len,args):
    proxy = get_proxy_model(tokenizer,num_token,max_len,args)
    # sigmoid = nn.Sigmoid()

    l2r = lambda x: x.clamp(min=0) / 1
    acq_fn = get_acq_fn()
    return acq_fn(proxy, l2r, args)


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

class Genetic_GFN_AL_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn_al"

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
        
        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])
        
        proxy = construct_proxy(voc, voc.vocab_size, max_len=200, args=config)
        proxy_dataset = Experience(voc, max_size=config['num_keep'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        best_scores = []

        assert config['random_action_prob'] * config['population_size'] == 0, "Cannot have both random actions and population size > 0"

        eps_noise = config['random_action_prob'] if config['population_size'] == 0 else 0.0

        # Random sample and pretrain
        # smiles = sample(config, Agent, oracle, voc, initial=True)
        # print("Initializing the datasets ...")
        
        
        prev_best = 0.
        experience = Experience(voc, max_size=config['num_keep'])  # gen_model_dataset
        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            # if len(self.oracle) > 6000:
            #     import pdb; pdb.set_trace()
            
            num_proxy_sample_iterations = config['num_proxy_init_sample_iterations'] if step == 0 else config['num_proxy_sample_iterations']
            for _ in range(num_proxy_sample_iterations):
                seqs, agent_likelihood, _ = Agent.sample(config['batch_size'], eps=eps_noise)
                # Remove duplicates, ie only consider unique seqs
                unique_idxs = unique(seqs)
                seqs = seqs[unique_idxs]
                agent_likelihood = agent_likelihood[unique_idxs]

                smiles = seq_to_smiles(seqs, voc)
                if config['valid_only']:
                    smiles = sanitize(smiles)
                start = perf_counter()
                score = np.array(self.oracle(smiles))
                # avg_time = (perf_counter()-start) / len(smiles)
                # print(avg_time)
                # if avg_time > 0.5:
                #     import pdb; pdb.set_trace()
                
                # Then add new experience
                new_experience = zip(smiles, score)
                proxy_dataset.add_experience(new_experience)

                if self.finish:
                    print('max oracle hit, abort ...... ')
                    break 

                if config['population_size'] and len(proxy_dataset) > config['population_size']:
                    # self.oracle.sort_buffer()
                    pop_smis, pop_scores = proxy_dataset.get_elems()

                    mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

                    for g in range(config['ga_generations']):
                        child_smis, child_n_atoms, pop_smis, pop_scores = ga_handler.query(
                                query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, 
                                rank_coefficient=config['rank_coefficient'], return_dist=False
                            )
                        
                        # tot_ga_results = pd.concat([tot_ga_results, pd.DataFrame(ga_results)])
                        start = perf_counter()
                        child_score = np.array(self.oracle(child_smis))
                        # print(f'GA {g}:', (perf_counter()-start) / len(smiles))

                        new_experience = zip(child_smis, child_score)
                        proxy_dataset.add_experience(new_experience)

                        mating_pool = (pop_smis+child_smis, pop_scores+child_score.tolist())
                
                if self.finish:
                    print('max oracle hit, abort ...... ')
                    break 
            
            if self.finish:
                break 

            print(len(proxy_dataset), 'train mols')
            proxy.model.train()
            proxy.fit(proxy_dataset)

            experience = Experience(voc, max_size=config['num_keep'])  # gen_model_dataset
            
            # Generative model training
            # print("Model training ...")
            pbar = tqdm(range(config['num_iterations']))
            pbar.set_description('Generative model training')
            for i in pbar:
                # Sample from Agent
                with torch.no_grad():
                    seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'], eps=eps_noise)

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
                
                # score = np.array(self.oracle(smiles))
                with torch.no_grad():
                    ys = proxy(seqs).view(-1).cpu().detach().numpy()

                # Then add new experience
                new_experience = zip(smiles, ys)
                experience.add_experience(new_experience)

                if config['population_size'] and len(experience) > config['population_size']:
                    # self.oracle.sort_buffer()
                    pop_smis, pop_scores = experience.get_elems()

                    mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

                    for g in range(config['ga_generations']):
                        child_smis, child_n_atoms, pop_smis, pop_scores, ga_results = ga_handler.query(
                                query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, 
                                rank_coefficient=config['rank_coefficient'], return_dist=True
                            )
                        
                        # tot_ga_results = pd.concat([tot_ga_results, pd.DataFrame(ga_results)])

                        # child_score = np.array(self.oracle(child_smis))
                        encoded = []
                        for i, smis in enumerate(child_smis):
                            try:
                                tokenized = voc.tokenize(smis)
                                encoded.append(Variable(voc.encode(tokenized)))
                            except:
                                pass
                            
                        if len(encoded) > 0:
                            encoded = MolData.collate_fn(encoded)

                            with torch.no_grad():
                                child_score = proxy(encoded).view(-1).cpu().detach().numpy()
                        
                            new_experience = zip(child_smis, child_score)
                            experience.add_experience(new_experience)

                            mating_pool = (pop_smis+child_smis, pop_scores+child_score.tolist())

                # Experience Replay
                # First sample
                avg_loss = 0.
                if config['experience_replay'] and len(experience) > config['experience_replay']:
                    for _ in range(config['experience_loop']):
                        num_samples = int(config['experience_replay'] * config['gamma'])
                        exp_seqs, exp_score = experience.rank_based_sample(num_samples, config['rank_coefficient'])
                        exp_seqs2, exp_score2 = proxy_dataset.rank_based_sample(config['experience_replay'] - num_samples, config['rank_coefficient'])
                        max_len = max(exp_seqs.size(1), exp_seqs2.size(1))
                        exp_seqs = torch.cat([exp_seqs, torch.zeros(exp_seqs.size(0), max(0, max_len - exp_seqs.size(1))).to(exp_seqs.device)], dim=1)
                        exp_seqs2 = torch.cat([exp_seqs2, torch.zeros(exp_seqs2.size(0), max(0, max_len - exp_seqs2.size(1))).to(exp_seqs2.device)], dim=1)
                        exp_seqs = torch.cat([exp_seqs, exp_seqs2])
                        # import pdb; pdb.set_trace()
                        exp_score = np.concatenate([exp_score, exp_score2])

                        exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                        prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                        reward = torch.tensor(exp_score).cuda()

                        exp_forward_flow = exp_agent_likelihood + log_z
                        exp_backward_flow = reward * config['beta']

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
                        pbar.set_postfix(loss = loss.item())

            # Random sample and pretrain
            # smiles, score = sample(config, Agent, oracle, voc, initial=False)
            # for _ in range(5):
            #     seqs, agent_likelihood, _ = Agent.sample(config['num_samples'], eps=eps_noise)
            #     # Remove duplicates, ie only consider unique seqs
            #     unique_idxs = unique(seqs)
            #     seqs = seqs[unique_idxs]
            #     agent_likelihood = agent_likelihood[unique_idxs]

            #     smiles = seq_to_smiles(seqs, voc)
            #     if config['valid_only']:
            #         smiles = sanitize(smiles)
            #     score = np.array(self.oracle(smiles))
                
            #     # Then add new experience
            #     new_experience = zip(smiles, score)
            #     proxy_dataset.add_experience(new_experience)

            #     if self.finish:
            #         print('max oracle hit, abort ...... ')
            #         break 

            #     if config['population_size'] and len(proxy_dataset) > config['population_size']:
            #         # self.oracle.sort_buffer()
            #         pop_smis, pop_scores = proxy_dataset.get_elems()

            #         mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

            #         for g in range(config['ga_generations']):
            #             child_smis, child_n_atoms, pop_smis, pop_scores = ga_handler.query(
            #                     query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, 
            #                     rank_coefficient=config['rank_coefficient'], return_dist=False
            #                 )
                        
            #             # tot_ga_results = pd.concat([tot_ga_results, pd.DataFrame(ga_results)])

            #             child_score = np.array(self.oracle(child_smis))

            #             new_experience = zip(child_smis, child_score)
            #             proxy_dataset.add_experience(new_experience)

            #             mating_pool = (pop_smis+child_smis, pop_scores+child_score.tolist())
            
            #     if self.finish:
            #         print('max oracle hit, abort ...... ')
            #         break 
            # if self.finish:
            #     print('max oracle hit, abort ...... ')
            #     break 
            
            # print(len(proxy_dataset), 'train mols')
            # proxy.model.train()
            # proxy.update(proxy_dataset)
            
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
                import pdb; pdb.set_trace()
                stuck_cnt += 1
                if stuck_cnt >= 10:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)

            step += 1
