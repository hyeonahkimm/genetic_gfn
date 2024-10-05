import os, sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb

import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

# rdkit
from rdkit import Chem, DataStructs

# from vocabulary import read_vocabulary
from utils import calc_fingerprints
from scoring_function import get_scores, int_div#, get_original_docking_scores

from model import RNN
from data_structs import Vocabulary, Experience
# from scoring_functions import get_scoring_function
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
# from vizard_logger import VizardLog
from joblib import Parallel
from graph_ga_expert import GeneticOperatorHandler


def sa_filter(smiles, scores, all_scores):
    filtered_smiles, filtered_scores, filtered_all_scores = [], [], []
    # import pdb; pdb.set_trace()
    for i, (_, _, sa) in enumerate(all_scores):
        if sa < 4.0:
            filtered_smiles.append(smiles[i])
            filtered_scores.append(scores[i])
            # filtered_seqs.append(seqs[i])
            filtered_all_scores.append(all_scores[i])

    return filtered_smiles, np.array(filtered_scores), np.stack(filtered_all_scores)


class Genetic_GFN_trainer():
    def __init__(self, logger, configs):
        self.writer = logger
        self.oracle = configs.oracle.strip()
        self.prior_path = configs.prior_path
        self.vocab_path = configs.vocab_path
        self.batch_size = configs.batch_size
        self.n_steps = configs.n_steps
        self.learning_rate = configs.learning_rate
        # experience replay
        self.memory = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps", "all_scores"])
        self.memory_size = configs.memory_size
        # Genetic search
        self.population_size = configs.population_size
        self.offspring_size = configs.offspring_size
        self.ga_generations = configs.ga_generations
        self.mutation_rate = configs.mutation_rate
        # Training
        self.experience_loop = configs.experience_loop
        self.experience_replay = configs.experience_replay
        self.lr_z = configs.lr_z
        self.beta = configs.beta
        self.rank_coefficient = configs.rank_coefficient
        self.kl_coefficient = configs.kl_coefficient
        self.wandb = configs.wandb
        self.run_name = configs.run_name

        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs.seed)
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def _memory_update(self, smiles, scores, all_scores):
        scores = list(scores)
        # seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]

        # fps_memory = list(self.memory["fps"])

        mean_coef = 0
        for i in range(len(smiles)):
            if scores[i] < 0:
                continue
            # canonicalized SMILES and fingerprints
            # fp, smiles_i = calc_fingerprints([smiles[i]])
            new_data = pd.DataFrame({"smiles": smiles[i], "scores": scores[i], "all_scores": [all_scores[i]]})
            # new_data = pd.DataFrame({"smiles": smiles[i], "scores": scores[i], "seqs": [seqs_list[i]], "all_scores": [all_scores[i]]})
            self.memory = pd.concat([self.memory, new_data], ignore_index=True, sort=False)

            # penalize similarity
            # if self.sim_penalize and len(fps_memory) > 0:
            #     sims = [DataStructs.FingerprintSimilarity(fp[0], x) for x in fps_memory]
            #     if np.sum(np.array(sims) >= self.sim_thres) > 20:
            #     	scores[i] = 0

        self.memory = self.memory.drop_duplicates(subset=["smiles"])
        self.memory = self.memory.sort_values('scores', ascending=False)
        self.memory = self.memory.reset_index(drop=True)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory.head(self.memory_size)

    def train(self):
        if not os.path.exists(f'outputs/{self.oracle}'):
            os.makedirs(f'outputs/{self.oracle}')

        voc = Vocabulary(init_from_file=self.vocab_path)

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if self.device.startswith('cuda'):
            Prior.rnn.load_state_dict(torch.load(self.prior_path))
            Agent.rnn.load_state_dict(torch.load(self.prior_path))
        else:
            Prior.rnn.load_state_dict(torch.load(self.prior_path, map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(self.prior_path, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        experience = Experience(voc, max_size=self.memory_size)

        log_z = torch.nn.Parameter(torch.tensor([5.]).cuda())
        optimizer = torch.optim.Adam([{'params': Agent.rnn.parameters(), 
                                        'lr': self.learning_rate},
                                    {'params': log_z, 
                                        'lr': self.lr_z}])
        ga_handler = GeneticOperatorHandler(mutation_rate=self.mutation_rate, 
                                            population_size=self.population_size)
        pool = Parallel(n_jobs=1)

        if 'mpo' in self.oracle:
            num_metric = 4
        else:
            num_metric = 2

        print("Model initialized, starting training...")

        for step in tqdm(range(self.n_steps)):
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample(self.batch_size)

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)

            # scores = np.array(self.score_smi(smiles))
            all_scores = np.array(get_scores(smiles, mode=self.oracle))
            # import pdb; pdb.set_trace()
            all_scores = all_scores.reshape(-1, num_metric)
            scores = all_scores[:, 0]  #np.clip(all_scores[:, 0], a_min=0., a_max=10.)
            # print(scores.max(), scores.mean())
            # print('docking score:', all_scores[:, 1].min(), all_scores[:, 1].mean())
            smiles, scores, all_scores = sa_filter(smiles, scores, all_scores[:, 1:])
            self._memory_update(smiles, scores, all_scores)
            
            new_experience = zip(smiles, scores)
            experience.add_experience(new_experience)

            # Genetic search
            if self.population_size and len(self.memory) > self.population_size:
                # mating_pool = (self.memory.smiles.tolist(), self.memory.scores.tolist())
                # import pdb; pdb.set_trace()
                # pop_smis, pop_scores = ga_handler.select_pop(self.memory.smiles.tolist(), self.memory.scores.tolist(), self.population_size, rank_coefficient=self.rank_coefficient)
                population = (self.memory.smiles.tolist(), self.memory.scores.tolist())  #(pop_smis, pop_scores)
                for g in range(self.ga_generations):
                    child_smis, child_n_atoms, pop_smis, pop_scores = ga_handler.query(
                            query_size=self.offspring_size, mating_pool=population, pool=pool, 
                            rank_coefficient=self.rank_coefficient, 
                        )

                    # child_score = np.array(self.oracle(child_smis))
                    child_all_scores = np.array(get_scores(child_smis, mode=self.oracle))
                    child_all_scores = child_all_scores.reshape(-1, num_metric)
                    child_score = child_all_scores[:, 0]
                    # import pdb; pdb.set_trace()
                    child_smis, child_score, child_all_scores = sa_filter(child_smis, child_score, child_all_scores[:, 1:])
                    self._memory_update(child_smis, child_score, child_all_scores)
                
                    new_experience = zip(child_smis, child_score)
                    experience.add_experience(new_experience)

                    population = (pop_smis+child_smis, pop_scores+child_score.tolist())

            # Experience Replay
            # First sample
            avg_loss = 0.
            if self.experience_replay and len(experience) > self.experience_replay:
                for _ in range(self.experience_loop):
                    exp_seqs, exp_score = experience.rank_based_sample(self.experience_replay, self.rank_coefficient)

                    exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                    prior_agent_likelihood, _ = Prior.likelihood(exp_seqs.long())

                    reward = torch.tensor(exp_score).cuda()

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = reward * self.beta
                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # KL penalty
                    loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                    loss += self.kl_coefficient*loss_p

                    # print(loss.item())
                    avg_loss += loss.item()/self.experience_loop

                    optimizer.zero_grad()
                    loss.backward()
                    # grad_norms = torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 1.0)
                    optimizer.step()

            # import pdb; pdb.set_trace()
            if len(self.memory) > 100:

                log = {'top-1': self.memory["scores"][0], 
                       'top-10': np.mean(np.array(self.memory["scores"][:10])),
                       'top-100': np.mean(np.array(self.memory["scores"][:100])),
                       'int_div': int_div(list(self.memory["smiles"][:100])),
                       'top-1 docking': self.memory["all_scores"][0][0],
                       'top-10 docking': np.stack(np.array(self.memory["all_scores"][:10]))[:, 0].mean(),
                       'top-100 docking': np.stack(np.array(self.memory["all_scores"][:100]))[:, 0].mean(),
                       'step_': step}
                
                if num_metric > 2:
                    log['top-1 qed'] = self.memory["all_scores"][0][1]
                    log['top-10 qed'] = np.stack(np.array(self.memory["all_scores"][:10]))[:, 1].mean()
                    log['top-100 qed'] = np.stack(np.array(self.memory["all_scores"][:100]))[:, 1].mean()
                    log['top-1 sa'] = self.memory["all_scores"][0][2]
                    log['top-10 sa'] = np.stack(np.array(self.memory["all_scores"][:10]))[:, 2].mean()
                    log['top-100 sa'] = np.stack(np.array(self.memory["all_scores"][:100]))[:, 2].mean()
                
                if self.wandb == 'online':
                    wandb.log(log)
                else:
                    print(f'top-1 score: {self.memory["scores"][0]}')
                    print(f'top-10 score: {np.mean(np.array(self.memory["scores"][:10]))}')
                    print(f'top-100 score: {np.mean(np.array(self.memory["scores"][:100]))}, diversity: {int_div(list(self.memory["smiles"][:100]))}')
                    print(f'top-1 docking score: {self.memory["all_scores"][0][0]}')
                    print(f'top-10 docking score: {np.mean(np.stack(np.array(self.memory["all_scores"][:10]))[:, 0])}')
                    print(f'top-100 docking score: {np.mean(np.stack(np.array(self.memory["all_scores"][:100]))[:, 0])}')
                
                self.writer.add_scalar('mean score in memory', np.mean(np.array(self.memory["scores"])), step)
                self.writer.add_scalar('top-1', self.memory["scores"][0], step)
                self.writer.add_scalar('top-10', np.mean(np.array(self.memory["scores"][:10])), step)
                self.writer.add_scalar('top-100', np.mean(np.array(self.memory["scores"][:100])), step)


            if (step + 1) % 20 == 0:
                # torch.save(agents[0].state_dict(), args.output_dir + f"QED_finetuned_{step+1}.pt")
                self.writer.add_scalar('top-100-div', int_div(list(self.memory["smiles"][:100])), step)
                self.writer.add_scalar('memory-div', int_div(list(self.memory["smiles"])), step)

            if (step + 1) % 50 == 0:
                self.memory.to_csv(f'outputs/{self.oracle}/genetic_gfn_{self.run_name}_{step+1}steps.csv')

        # docking, qed, sa, mpo_score = get_original_docking_scores(self.memory["smiles"][:100], self.oracle)

        self.memory.to_csv(f'outputs/{self.oracle}/genetic_gfn_{self.run_name}_{self.n_steps}steps.csv')
        print(f'top-1 score: {self.memory["scores"][0]}')
        print(f'top-10 score: {np.mean(np.array(self.memory["scores"][:10]))}')
        print(f'top-100 score: {np.mean(np.array(self.memory["scores"][:100]))}, diversity: {int_div(list(self.memory["smiles"][:100]))}')


class REINVENT_trainer():

    def __init__(self, logger, configs):
        self.writer = logger
        self.oracle = configs.oracle.strip()
        self.prior_path = configs.prior_path
        self.vocab_path = configs.vocab_path
        # self.voc = read_vocabulary(configs.vocab_path)
        self.batch_size = configs.batch_size
        self.n_steps = configs.n_steps
        self.learning_rate = configs.learning_rate
        self.sigma = configs.sigma
        # experience replay
        self.memory = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps", "all_scores"])
        self.memory_size = configs.memory_size
        self.experience_replay = configs.experience_replay

        self.wandb = configs.wandb

        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs.seed)
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def _memory_update(self, smiles, scores, seqs, all_scores):
        scores = list(scores)
        seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]

        fps_memory = list(self.memory["fps"])

        mean_coef = 0
        for i in range(len(smiles)):
            if scores[i] < 0:
                continue
            # canonicalized SMILES and fingerprints
            # fp, smiles_i = calc_fingerprints([smiles[i]])
            new_data = pd.DataFrame({"smiles": smiles[i], "scores": scores[i], "seqs": [seqs_list[i]], "all_scores": [all_scores[i]]})
            self.memory = pd.concat([self.memory, new_data], ignore_index=True, sort=False)

            # penalize similarity
            # if self.sim_penalize and len(fps_memory) > 0:
            #     sims = [DataStructs.FingerprintSimilarity(fp[0], x) for x in fps_memory]
            #     if np.sum(np.array(sims) >= self.sim_thres) > 20:
            #     	scores[i] = 0

        self.memory = self.memory.drop_duplicates(subset=["smiles"])
        self.memory = self.memory.sort_values('scores', ascending=False)
        self.memory = self.memory.reset_index(drop=True)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory.head(self.memory_size)

        # # experience replay
        # if self.replay > 0:
        #     s = min(len(self.memory), self.replay)
        #     experience = self.memory.head(5 * self.replay).sample(s)
        #     experience = experience.reset_index(drop=True)
        #     smiles += list(experience["smiles"])
        #     scores += list(experience["scores"])
        #     for index in experience.index:
        #         seqs = torch.cat((seqs, torch.tensor(experience.loc[index, "seqs"], dtype=torch.long).view(1, -1).cuda()), dim=0)

        # return smiles, np.array(scores), seqs


    def train(self):
        if not os.path.exists(f'outputs/{self.oracle}'):
            os.makedirs(f'outputs/{self.oracle}')

        voc = Vocabulary(init_from_file=self.vocab_path)

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if self.device.startswith('cuda'):
            Prior.rnn.load_state_dict(torch.load(self.prior_path))
            Agent.rnn.load_state_dict(torch.load(self.prior_path))
        else:
            Prior.rnn.load_state_dict(torch.load(self.prior_path, map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(self.prior_path, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        experience = Experience(voc, max_size=self.memory_size)
        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=self.learning_rate)

        if 'mpo' in self.oracle:
            num_metric = 4
        else:
            num_metric = 2

        print("Model initialized, starting training...")

        for step in tqdm(range(self.n_steps)):
            # samples, seqs, _ = sample_SMILES(agents[i], self.voc, n_mols=self.batch_size)
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample(self.batch_size)

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)

            # scores = np.array(self.score_smi(smiles))
            all_scores = np.array(get_scores(smiles, mode=self.oracle))
            # import pdb; pdb.set_trace()
            all_scores = all_scores.reshape(-1, num_metric)
            scores = all_scores[:, 0]  #np.clip(all_scores[:, 0], a_min=0., a_max=10.)
            # print(scores.max(), scores.mean())
            # print('docking score:', all_scores[:, 1].min(), all_scores[:, 1].mean())
            self._memory_update(smiles, scores, seqs, all_scores[:, 1:])

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood + self.sigma * Variable(scores).float()
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            # First sample
            if self.experience_replay and len(experience) > self.experience_replay:
                exp_seqs, exp_score = experience.sample(self.experience_replay)
                exp_prior_likelihood, _ = Prior.likelihood(exp_seqs.long())
                exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
                # import pdb; pdb.set_trace()
                exp_augmented_likelihood = exp_prior_likelihood + self.sigma *  Variable(exp_score).float()
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

            # Then add new experience
            prior_likelihood = prior_likelihood.data.cpu().numpy()
            new_experience = zip(smiles, scores, prior_likelihood)
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

            # import pdb; pdb.set_trace()
            if len(self.memory) > 100:

                log = {'top-1': self.memory["scores"][0], 
                       'top-10': np.mean(np.array(self.memory["scores"][:10])),
                       'top-100': np.mean(np.array(self.memory["scores"][:100])),
                       'int_div': int_div(list(self.memory["smiles"][:100])),
                       'top-1 docking': self.memory["all_scores"][0][0],
                       'top-10 docking': np.stack(np.array(self.memory["all_scores"][:10]))[:, 0].mean(),
                       'top-100 docking': np.stack(np.array(self.memory["all_scores"][:100]))[:, 0].mean(),
                       'step_': step}
                
                if num_metric > 2:
                    log['top-1 qed'] = self.memory["all_scores"][0][1]
                    log['top-10 qed'] = np.stack(np.array(self.memory["all_scores"][:10]))[:, 1].mean()
                    log['top-100 qed'] = np.stack(np.array(self.memory["all_scores"][:100]))[:, 1].mean()
                    log['top-1 sa'] = self.memory["all_scores"][0][2]
                    log['top-10 sa'] = np.stack(np.array(self.memory["all_scores"][:10]))[:, 2].mean()
                    log['top-100 sa'] = np.stack(np.array(self.memory["all_scores"][:100]))[:, 2].mean()
                
                if self.wandb == 'online':
                    wandb.log(log)
                else:
                    print(f'top-1 score: {self.memory["scores"][0]}')
                    print(f'top-10 score: {np.mean(np.array(self.memory["scores"][:10]))}')
                    print(f'top-100 score: {np.mean(np.array(self.memory["scores"][:100]))}, diversity: {int_div(list(self.memory["smiles"][:100]))}')
                    print(f'top-1 docking score: {self.memory["all_scores"][0][0]}')
                    print(f'top-10 docking score: {np.mean(np.stack(np.array(self.memory["all_scores"][:10]))[:, 0])}')
                    print(f'top-100 docking score: {np.mean(np.stack(np.array(self.memory["all_scores"][:100]))[:, 0])}')
                
                self.writer.add_scalar('mean score in memory', np.mean(np.array(self.memory["scores"])), step)
                self.writer.add_scalar('top-1', self.memory["scores"][0], step)
                self.writer.add_scalar('top-10', np.mean(np.array(self.memory["scores"][:10])), step)
                self.writer.add_scalar('top-100', np.mean(np.array(self.memory["scores"][:100])), step)


            if (step + 1) % 20 == 0:
                # torch.save(agents[0].state_dict(), args.output_dir + f"QED_finetuned_{step+1}.pt")
                self.writer.add_scalar('top-100-div', int_div(list(self.memory["smiles"][:100])), step)
                self.writer.add_scalar('memory-div', int_div(list(self.memory["smiles"])), step)

            if (step + 1) % 50 == 0:
                self.memory.to_csv(f'outputs/{self.oracle}/reinvent_{step+1}steps.csv')

        # docking, qed, sa, mpo_score = get_original_docking_scores(self.memory["smiles"][:100], self.oracle)

        self.memory.to_csv(f'outputs/{self.oracle}/reinvent_{self.n_steps}steps.csv')
        print(f'top-1 score: {self.memory["scores"][0]}')
        print(f'top-10 score: {np.mean(np.array(self.memory["scores"][:10]))}')
        print(f'top-100 score: {np.mean(np.array(self.memory["scores"][:100]))}, diversity: {int_div(list(self.memory["smiles"][:100]))}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='reinvent')
    parser.add_argument('--run_name', type=str, default="default")
    parser.add_argument('--oracle', type=str, default="JNK3")
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=112)
    parser.add_argument('--sigma', type=float, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--population_size', type=int, default=64)
    parser.add_argument('--offspring_size', type=int, default=8)
    parser.add_argument('--ga_generations', type=int, default=2)
    parser.add_argument('--mutation_rate', type=float, default=0.01)
    parser.add_argument('--experience_loop', type=int, default=8)
    parser.add_argument('--experience_replay', type=int, default=128)
    parser.add_argument('--lr_z', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=50)
    parser.add_argument('--rank_coefficient', type=float, default=0.01)
    parser.add_argument('--kl_coefficient', type=float, default=0.01)
    # parser.add_argument('--sim_penalize', type=bool, default=False)
    # parser.add_argument('--sim_thres', type=float, default=0.7)
    parser.add_argument('--prior_path', type=str, default="genetic_gfn/data/Prior.ckpt")
    parser.add_argument('--vocab_path', type=str, default="genetic_gfn/data/Voc")
    parser.add_argument('--output_dir', type=str, default="log/")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    print(args)

    # set_seed(42)

    writer = SummaryWriter(args.output_dir + f"{args.oracle}/{args.method}_{args.run_name}/")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    writer.add_text("configs", str(args))

    if args.wandb == 'online':
        wandb.init(project='sars_covid', group=args.oracle, config=args, reinit=True)
        if args.run_name == "":
            wandb.run.name = args.oracle + "_" + args.method + "_" + str(args.seed) + "_" + wandb.run.id
        else:
            wandb.run.name = args.oracle + "_" + args.method + "_" + args.run_name + "_" + str(args.seed) + "_" + wandb.run.id

    args.run_name += f'_{args.seed}'
    if args.method == "reinvent":
        args.population_size = 0
        RL_trainer = REINVENT_trainer(logger=writer, configs=args)
    elif args.method == "genetic_gfn":
        assert args.population_size > 0
        RL_trainer = Genetic_GFN_trainer(logger=writer, configs=args)
    else:
        raise ValueError("Unrecognized method name.")
    RL_trainer.train()

    writer.close()
    