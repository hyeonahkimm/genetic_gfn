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

from joblib import Parallel
from graph_ga_expert import GeneticOperatorHandler
# from smiles_ga_expert import GeneticOperatorHandler as SmilesGA


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
            # if not valid_only:
            #     canonicalized.append(s)
    return canonicalized


class REINVENT_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent_ga"

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
        # ga_experience = Experience(voc, max_size=config['num_keep'])
        # experience = MaxRewardPriorityQueue()

        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_max = 0.
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
            # seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'], temp = 1. + float(config['dynamic_temp']) * min(stuck_cnt, 10) * 0.1)

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)
            if config['valid_only']:
                smiles = sanitize(smiles)
            # if len(smiles) == 0:
            #     print('no valid smiles, use invalid too')
            #     smiles = seq_to_smiles(seqs, voc)
            #     smiles = canonicalize(smiles)
            
            score = np.array(self.oracle(smiles))

            # delta_score = np.clip(score.max() - prev_max, a_min = 0, a_max = 0.5)
            # if prev_max < score.max():
            #     prev_max = score.max()
            #     stuck_cnt = 0
            #     print('NN:', score.max(), score.mean(), score.std(), len(score))
            # else:
            #     stuck_cnt += 1
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

            # forward_flow = agent_likelihood + log_z
            # backward_flow = Variable(score).float() * config['beta']
            # loss = torch.pow(forward_flow-backward_flow, 2).mean()


            # Then add new experience
            # prior_likelihood = prior_likelihood.data.cpu().numpy()
            # new_experience = zip(smiles, score, prior_likelihood)
            
            new_experience = zip(smiles, score)
            experience.add_experience(new_experience)

            # experience.add_list(seqs=seqs, smis=smiles, scores=score)
            # experience.squeeze_by_kth(k=config['num_keep'])

            if config['population_size'] and len(self.oracle) > config['population_size']:
                if config['ga_sample_from'] == 'experience':
                    pop_smis, pop_scores = experience.get_elems()
                elif config['ga_sample_from'] == 'sample':
                    pop_smis, pop_scores = smiles, score
                else:
                    self.oracle.sort_buffer()
                    pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                # print(list(self.oracle.mol_buffer))
                mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

                for g in range(config['ga_generations']):
                    child_smis, child_n_atoms, pop_smis, pop_scores = ga_handler.query(
                            # query_size=50, mating_pool=mating_pool, pool=pool, return_pop=True
                            query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, 
                            rank_coefficient=config['rank_coefficient'], 
                            blended=config['blended_ga'],
                            # mutation_rate = config['mutation_rate'] + float(config['dynamic_temp']) * delta_score
                            low_score_ratio = config['low_score_ratio'],
                            canonicalize=config['canonicalize']
                        )

                    # child_smis = list(set(child_smis))
                    child_score = np.array(self.oracle(child_smis))
                    
                    # high_smis, high_score = [], []
                    # for idx, s in enumerate(score):
                    #     if s > prev_max:
                    #         # new_experience = zip(smis[idx], s)
                    #         high_smis.append(smis[idx])
                    #         high_score.append(s)
                    
                    if not config['canonicalize']:
                        new_experience = zip(child_smis, child_score, child_n_atoms)
                    else:
                        new_experience = zip(child_smis, child_score)
                    experience.add_experience(new_experience)

                    mating_pool = (pop_smis+child_smis, pop_scores+child_score.tolist())

                    # delta_score = np.clip(score.max() - prev_max, a_min = 0, a_max = 0.5)
                    # if prev_max < score.max():
                    #     prev_max = score.max()
                    #     stuck_cnt = 0
                    # else:
                    #     stuck_cnt += 1
                    # print('GA' + str(g+1) + ':', child_score.max(), child_score.mean(), child_score.std(), len(child_smis))

                    if self.finish:
                        print('max oracle hit')
                        break
                
                # if config['ga_generations'] > 1:
                #     smis, score = ga_handler.get_final_population(mating_pool)
                # new_experience = zip(mating_pool[0], mating_pool[1])
                # new_experience = zip(child_smis, child_score)
                # experience.add_experience(new_experience)

            
            # Experience Replay
            # First sample
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    if config['rank_coefficient'] >= 1:
                        # exp_seqs, exp_score = experience.quantile_uniform_sample(config['experience_replay'], 0.001)
                        exp_seqs, exp_score, _ = experience.rank_based_sample(config['experience_replay'], config['train_rank_coefficient'])
                    elif config['rank_coefficient'] > 0:
                        exp_seqs, exp_score, exp_pb = experience.rank_based_sample(config['experience_replay'], config['rank_coefficient'])
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
                    elif config['penalty'] == 'prior':
                        # print(prior_agent_likelihood.mean())
                        # print((exp_agent_likelihood - prior_agent_likelihood).mean())
                        reward += 0.001 * prior_agent_likelihood

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * config['beta']
                    # if not config['canonicalize']:
                    #     # print(exp_pb)
                    #     exp_backward_flow += torch.tensor(np.log(1/exp_pb)).cuda()  # approximated uniform pb
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
                        loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                        # print(loss.item(), loss_p.item())
                        # print(loss_p.item(), 1e-2 * torch.nn.functional.mse_loss(exp_agent_likelihood, prior_agent_likelihood).item(), 1e3 * (1 / exp_agent_likelihood).mean().item())
                        loss += config['kl_coefficient']*loss_p
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

