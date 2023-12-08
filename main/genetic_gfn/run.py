import os
path_here = os.path.dirname(os.path.realpath(__file__))

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from joblib import Parallel

from main.optimizer import BaseOptimizer
from runner.gegl_trainer import GeneticExpertGuidedLearningTrainer
from runner.guacamol_generator import GeneticExpertGuidedLearningGenerator
from model.neural_apprentice import SmilesGenerator, SmilesGeneratorHandler
from model.graph_ga_expert import GeneticOperatorHandler
from util.storage.priority_queue import MaxRewardPriorityQueue
from util.storage.recorder import Recorder
from util.chemistry.benchmarks import load_benchmark
from util.smiles.char_dict import SmilesCharDictionary


class GeneticGFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        device = torch.device(0)

        char_dict = SmilesCharDictionary(dataset=config['dataset'], max_smi_len=config['max_smiles_length'])

        # Prepare max-reward priority queues
        apprentice_storage = MaxRewardPriorityQueue()
        expert_storage = MaxRewardPriorityQueue()

        # Prepare neural apprentice (we use the weights pretrained on existing dataset)
        apprentice = SmilesGenerator.load(load_dir=config['apprentice_load_dir'])
        apprentice = apprentice.to(device)

        if config['use_tb_loss']:
            log_z = torch.nn.Parameter(torch.tensor([5.], device=device))
            apprentice_optimizer = Adam([{'params': apprentice.parameters(), 
                                         'lr': config['learning_rate']},
                                        {'params': log_z, 
                                         'lr': config['lr_z']}])
        else:
            log_z = None
            apprentice_optimizer = Adam(apprentice.parameters(), lr=config['learning_rate'])
        
        apprentice_handler = SmilesGeneratorHandler(
            model=apprentice,
            optimizer=apprentice_optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=config['max_sampling_batch_size'],
            log_z=log_z,
        )
        apprentice.train()

        # Prepare genetic expert
        expert_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], population_size=config['expert_sampling_batch_size'], rank_based=config['rank_based'])

        pool = Parallel(n_jobs=config['num_jobs'])

        step = 0
        patience = 0

        while True:
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            # update_storage_by_apprentice
            with torch.no_grad():
                apprentice_handler.model.eval()
                smis, _, _, _ = apprentice_handler.sample(
                    num_samples=config['apprentice_sampling_batch_size'], device=device
                )
            
            smis = list(set(smis))
            k = min((self.oracle.max_oracle_calls - len(self.oracle)), len(smis))
            smis = smis[:k]

            score = np.array(self.oracle(smis))
            # div = self.oracle.diversity_evaluator(smis)

            if self.finish:
                print('max oracle hit')
                break
            
            apprentice_storage.add_list(smis=smis, scores=score)
            apprentice_storage.squeeze_by_kth(k=config['num_keep'])

            # update_storage_by_expert
            if config['sample_from_storage']:
                expert_smis, expert_scores = apprentice_storage.sample_batch(
                    config['expert_sampling_batch_size']
                )
            else:
                apprentice_smis, apprentice_scores = apprentice_storage.get_elems()
                if len(expert_storage) > 0:
                    expert_smis, expert_scores = expert_storage.get_elems()
                    expert_smis += apprentice_smis
                    expert_scores += apprentice_scores
                else:
                    expert_smis = apprentice_smis
                    expert_scores = apprentice_scores
            
            mating_pool = (expert_smis, expert_scores)

            for gen in range(config['ga_generation']):

                if config['ga_blended']:
                    smis, population_mol, population_scores = expert_handler.blended_query(
                        query_size=config['ga_offspring_size'], mating_pool=mating_pool, pool=pool, rank_based=(gen==0), return_pop=True
                    )
                else:
                    smis, population_mol, population_scores = expert_handler.query(
                        # query_size=50, mating_pool=mating_pool, pool=pool, return_pop=True
                        query_size=config['ga_offspring_size'], mating_pool=mating_pool, pool=pool, rank_based=True, return_pop=True
                    )

                smis = list(set(smis))
                k = min((self.oracle.max_oracle_calls - len(self.oracle)), len(smis))
                smis = smis[:k]
                score = np.array(self.oracle(smis))

                mating_pool = (population_mol + smis, population_scores + score.tolist())

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

            if config['ga_generation'] > 1:
                smis, score = expert_handler.get_final_population(mating_pool, rank_based=False)

            expert_storage.add_list(smis=smis, scores=score)
            expert_storage.squeeze_by_kth(k=config['num_keep'])

            # train_apprentice_step
            avg_loss = 0.0
            apprentice_smis, apprentice_scores = apprentice_storage.get_elems()
            expert_smis, expert_scores = expert_storage.get_elems()
            total_smis = list(set(apprentice_smis + expert_smis))
            total = [(smiles, score) for smiles, score in zip(apprentice_smis + expert_smis, apprentice_scores + expert_scores)]

            apprentice_handler.model.train()
            for _ in range(config['num_apprentice_training_steps']):
                if config['use_tb_loss']:
                    # sampled_smis, sampled_score = expert_handler.get_final_population((apprentice_smis + expert_smis, apprentice_scores + expert_scores), rank_based=True)
                    # smis_scores = [(smiles, score) for smiles, score in zip(sampled_smis, sampled_score)]
                    smis_scores = random.choices(population=total, k=config['apprentice_training_batch_size'])
                    loss = apprentice_handler.train_tb(smis_scores=smis_scores, device=device, beta=config['beta'])
                    # loss = 0.
                    # training_steps = len(total) // config['apprentice_training_batch_size']
                    # random.shuffle(total)
                    # for step in range(int(training_steps)):
                    #     start_idx = step * config['apprentice_training_batch_size']
                    #     smis_scores = total[start_idx : start_idx + config['apprentice_training_batch_size']]
                    #     # smis_scores = random.choices(population=total, k=config['apprentice_training_batch_size']) 
                    #     loss += apprentice_handler.train_tb(smis_scores=smis_scores, device=device, beta=config['beta']) / training_steps
                else:
                    smis = random.choices(population=total_smis, k=config['apprentice_training_batch_size'])
                    loss = apprentice_handler.train_on_batch(smis=smis, device=device)

                avg_loss += loss / config['num_apprentice_training_steps']

            fit_size = len(total_smis)

            step += 1


