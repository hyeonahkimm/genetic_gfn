#!/usr/bin/env python

import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
from shutil import copyfile

from model import RNN
from data_structs import Vocabulary, Experience
# from scoring_functions import get_scoring_function
from scoring_function import get_scores, int_div
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
# from vizard_logger import VizardLog


def normalize_score(current_fitness, all_fitness, threshold = 0.7):
    all_fitness = np.array(all_fitness)[~np.isnan(all_fitness)]
    avg = np.mean(all_fitness)
    ceil = np.max(all_fitness)

    # sigmoid scaling
    threshold = 0.99 if threshold > 0.99 else threshold
    slope = -1.0 / (ceil - avg) * np.log( 2.0 / (threshold + 1.0 ) - 1.0)
    scaled = 2.0 / (1.0 + np.exp(-slope*(current_fitness - avg))) - 1.0
    
    # invalid smiles do not affect the agent, reward is 0
    scaled[np.isnan(scaled)] = 0.0
    return scaled

def train_agent(restore_prior_from='data/Prior.ckpt',
                restore_agent_from='data/Prior.ckpt',
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=3000,
                num_processes=0, sigma=60,
                experience_replay=0,
                seed=0):

    voc = Vocabulary(init_from_file="genetic_gfn/data/Voc")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)

    # Scoring_function
    # scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
    #                                         **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)
    
    # collect all the results
    results = pd.DataFrame({'smiles': [], 'fitness': [], 'score': [], 'generation': []})
    best_results = pd.DataFrame({'smiles': [], 'fitness': [], 'score': [], 'generation': []})

    print("Model initialized, starting training...")

    for step in range(n_steps):

        # Current generation
        collector = {'smiles': [], 'fitness': [], 'score': [], 'generation': []}

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        # print(smiles)
        # import pdb; pdb.set_trace()
        # fitness = scoring_function(smiles)
        fitness = get_scores(smiles, mode=scoring_function.strip())
        collector['smiles'] = smiles
        collector['fitness'] = fitness
        collector['generation'] = [step]*len(unique_idxs)

        # normalize using sigmoid
        # if step == 0:
        #     score = normalize_score(fitness, collector['fitness'])     
        # else:
        #     score = normalize_score(fitness, results['fitness'].tolist())
        score = np.array(fitness)
        # score[np.isnan(score)] = 0.0
        collector['score'] = score
        collector = pd.DataFrame(collector)

        results = pd.concat([results, collector])

        best = collector.nlargest(1, 'fitness')
        best_fitness = best['fitness'].values[0]
        best_smiles = best['smiles'].values[0]

        best = results.nlargest(1, 'fitness')
        best_fitness_all = best['fitness'].values[0]
        best_smiles_all = best['smiles'].values[0]
        best['generation'] = step
        best_results = pd.concat([best_results, best])

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > experience_replay:
            exp_seqs, exp_score = experience.sample(experience_replay)
            exp_prior_likelihood, _ = Prior.likelihood(exp_seqs.long())
            exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
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

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        # print("  Agent    Prior   Target   Score   Fitness             SMILES")
        # for i in range(10):
        #     print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}   {:6.2f}     {}".format(agent_likelihood[i],
        #                                                                prior_likelihood[i],
        #                                                                augmented_likelihood[i],
        #                                                                score[i],
        #                                                                fitness[i],
        #                                                                smiles[i]))
        print(f"Best in generation: {best_fitness:6.2f}   {best_smiles}")
        print(f"Best in general:    {best_fitness_all:6.2f}   {best_smiles_all}")


    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir)
    copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    results = results.reset_index(drop=True)
    best_results = best_results.reset_index(drop=True)

    results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    best_results.to_csv(os.path.join(save_dir, 'best_results.csv'), index=False)
    sns.lineplot(data=best_results, x='generation', y='fitness')
    plt.savefig(os.path.join(save_dir, 'trace.png'))

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))


if __name__ == "__main__":
    # train_agent()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--oracle', type=str, default="JNK3")
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--experience_replay', type=int, default=5)
    # parser.add_argument('--sim_penalize', type=bool, default=False)
    # parser.add_argument('--sim_thres', type=float, default=0.7)
    # parser.add_argument('--prior_path', type=str, default="reinvent/data/Prior.ckpt")
    # parser.add_argument('--vocab_path', type=str, default="data/vocab.txt")
    # parser.add_argument('--output_dir', type=str, default="log/")
    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    print(args)

    train_agent(
        restore_prior_from='genetic_gfn/data/Prior.ckpt',
        restore_agent_from='genetic_gfn/data/Prior.ckpt',
        scoring_function=args.oracle,
        # scoring_function_kwargs={'task':task_config['score'], 'obj_idx': task_config['obj_idx'], 'invalid_score': task_config['invalid_score']},
        batch_size = args.batch_size,
        n_steps = args.n_steps,
        # num_processes = -1,
    )
    