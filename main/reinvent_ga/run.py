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

from joblib import Parallel
from graph_ga_expert import GeneticOperatorHandler


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
        # experience = MaxRewardPriorityQueue()

        # Prepare genetic expert
        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0

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

            # print('NN:', score.max(), score.mean(), len(score))

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
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                # print(list(self.oracle.mol_buffer))
                mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

                for g in range(config['ga_generations']):

                    smis, pop_smis, pop_scores = ga_handler.query(
                            # query_size=50, mating_pool=mating_pool, pool=pool, return_pop=True
                            query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, rank_coefficient=config['rank_coefficient'], blended=config['blended_ga']
                        )

                    smis = list(set(smis))
                    score = np.array(self.oracle(smis))

                    mating_pool = (pop_smis+smis, pop_scores+score.tolist())

                    # print('GA' + str(g+1) + ':', score.max(), score.mean(), len(smis))

                    if self.finish:
                        print('max oracle hit')
                        break
                
                if config['ga_generations'] > 1:
                    smis, score = ga_handler.get_final_population(mating_pool)
                new_experience = zip(smis, score)
                experience.add_experience(new_experience)

            
            # Experience Replay
            # First sample
            avg_loss = 0.
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                for _ in range(config['experience_loop']):
                    # exp_seqs, exp_score = experience.sample(config['experience_replay'])
                    exp_seqs, exp_score = experience.rank_based_sample(config['experience_replay'])
                    # exp_seqs, exp_smis, exp_score = experience.sample_batch(config['experience_replay']) 
                    # tokenized = [voc.tokenize(smile) for smile in smiles]
                    # encoded = [Variable(voc.encode(tokenized_i)) for tokenized_i in tokenized]
                    # encoded = MolData.collate_fn(encoded)
                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    # prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())
                    # exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                    # exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                    # loss = torch.cat((loss, exp_loss), 0)
                    # agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = torch.tensor(exp_score).cuda() * config['beta']
                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # Add regularizer that penalizes high likelihood for the entire sequence
                    if config['penalty']:
                        loss_p = - (1 / exp_agent_likelihood).mean()
                        # print('penalty:', loss_p.item())
                        loss += 1e3 * loss_p

                    # print(loss.item())
                    avg_loss += loss.item()/config['experience_loop']

                    optimizer.zero_grad()
                    loss.backward()
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

