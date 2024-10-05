import torch
import numpy as np
import pandas as pd
import time
import os
import sys
import argparse

# path_here = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(path_here)
# sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))


# from vizard_logger import VizardLog
from tqdm import tqdm 

from rdkit import Chem
from botorch.utils.multi_objective.hypervolume import Hypervolume
from torch.distributions.dirichlet import Dirichlet

from genetic_gfn.model import RNN
from genetic_gfn.data_structs import Vocabulary, Experience
from genetic_gfn.utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from oracle.oracle import Oracle
from utils.metrics import compute_success, compute_diversity, compute_novelty
from dataset import Dataset


# def train_agent(restore_prior_from='data/Prior.ckpt',
#                 restore_agent_from='data/Prior.ckpt',
#                 scoring_function='tanimoto',
#                 scoring_function_kwargs=None,
#                 save_dir=None, learning_rate=0.0005,
#                 batch_size=64, n_steps=3000,
#                 num_processes=0, sigma=60,
#                 experience_replay=0):

def train_agent(args):
    path_here = os.path.dirname(os.path.realpath(__file__))
    voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))
    
    oracle = Oracle(args)
    args.n_objectives = len(args.objectives)
    hypervolume = Hypervolume(ref_point=torch.zeros(args.n_objectives).to(args.device))
    # mol_buffer = {}a
    bpath = "./data/blocks_105.json"
    # dataset = Dataset(args, bpath, oracle, args.device)
    
    path_here = os.path.dirname(os.path.realpath(__file__))
    restore_prior_from = os.path.join(path_here, args.restore_prior_from)
    restore_agent_from = restore_prior_from

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # logger = VizardLog('data/logs')

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0005)

    # Scoring_function
    # scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
    #                                         **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    # logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    # logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    # logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    # logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    # logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")

    # Information for the logger
    step_score = [[], []]
    score_succ = {'gsk3b': 0.5, 'jnk3': 0.5, 'drd2': 0.5, 
                  'chemprop_sars': 0.5, 'chemprop_hiv': 0.5, "seh": 0.5,
                  'qed': 0.6, 'sa': 0.67}
    
    
    print("Model initialized, starting training...")

    for step in tqdm(range(args.n_steps)):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(args.batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        
        picked_mols, picked_idx = [], []
        for i, s in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    picked_mols.append(mol)
                    picked_idx.append(i)
            except:
                pass
        
        # import pdb; pdb.set_trace()
        scores_dict = oracle.batch_get_scores(picked_mols)
        # pdb.set_trace()
        valid_scores = np.array(pd.DataFrame.from_dict(scores_dict).values)
        
        # weights = Dirichlet(torch.tensor(args.alpha_vector)*args.alpha).sample_n(1).to(args.device) #* sample weights per batch, seem better
        weights = np.array(args.alpha_vector)  #* fixed weights
        reward = np.matmul(valid_scores, weights.reshape(-1, 1))
        # volume = hypervolume.compute(valid_scores)
        score = np.zeros(len(smiles))
        score[picked_idx] = reward.reshape(-1)
        
        # for i in range(len(picked_mols)):
        #     picked_mols[i].score = scores_dict[i]

        #     # success/diversity/novelty is computed among the top mols.
        #     success, positive_mols = compute_success(
        #         picked_mols, scores_dict, args.objectives, score_succ)
        #     succ_diversity = compute_diversity(positive_mols)
        #     if ref_mols:
        #         novelty = compute_novelty(positive_mols, ref_mols)
        #     else:
        #         novelty = 1.

        #     mo_metrics = {'success': success, 'novelty': novelty,
        #                 'succ_diversity': succ_diversity, }

        #     picked_smis = [(scores[i], picked_mols[i].score, smiles[i])
        #                 for i in range(len(raw_rewards))]
        #     print(mo_metrics)
        #     return (picked_mols, scores_dict, picked_smis), batch_metrics, mo_metrics
        
        # score = np.array(scoring_function(smiles))
        # score = scoring_function(smiles)
        # if scoring_function.finish:
        #     print('max oracle hit')
        #     break 
        # else:
        #     print('----- not hit ------ ', len(scoring_function.mol_buffer))

        # print("prior_likelihood", type(prior_likelihood), ) ### 
        # print("score", score, type(score), ) ### np.array 
        # print(score.dtype) #### float64
        # print(Variable(score).shape)
        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood.float() + args.sigma * Variable(score).float()
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if args.experience_replay and len(experience) > args.experience_replay:
            exp_seqs, exp_score = experience.sample(args.experience_replay)
            exp_prior_likelihood, _ = Prior.likelihood(exp_seqs.long())
            exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
            # import pdb; pdb.set_trace()
            exp_augmented_likelihood = exp_prior_likelihood + args.sigma * Variable(exp_score).float()
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
        time_left = (time_elapsed * ((args.n_steps - step) / (step + 1)))
        
        print(score.mean(), score.min(), score.max())
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(5):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))

        # Log some weights
        # logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        # logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        # logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        # logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        # logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        # logger.log("\n".join([smiles + "\t" + str(round(score, 2)) for smiles, score in zip \
        #                     (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        # logger.log(np.array(step_score), "Scores")
    
    import pdb; pdb.set_trace()


    # print('train_agent', scoring_function.mol_buffer)
    # return scoring_function.mol_buffer  


# if __name__ == "__main__":
#     train_agent()
