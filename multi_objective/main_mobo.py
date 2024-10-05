from collections import defaultdict
import random
import os
import re
from dataset import Dataset
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from oracle.oracle import Oracle
from proxy import get_proxy
from generator import TBGFlowNet, FMGFlowNet, MOReinforce
from utils.utils import set_random_seed
from utils.metrics import compute_success, compute_diversity, compute_novelty, compute_correlation, circle_points
from utils.logging import get_logger
from datetime import datetime
import argparse
import json
import time
import threading
import pdb
import pickle
import gzip
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch
from torch.distributions.dirichlet import Dirichlet
import pandas as pd
import numpy as np
from main import RolloutWorker, get_test_mols
from pymoo.util.ref_dirs import get_reference_directions
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("--run", default=0, help="run", type=int)
    parser.add_argument('--save', action='store_true',
                        default=False, help='Save model.')
    parser.add_argument('--debug',action='store_true',
                        default=False, help='debug mode, no multi thread')
    parser.add_argument("--enable_tensorboard",
                        action='store_true', default=False)
    parser.add_argument("--log_dir", default='runs/mobo')
    parser.add_argument("--include_nblocks", default=False)
    parser.add_argument("--num_init_examples", default=200, type=int)
    parser.add_argument("--num_outer_loop_iters", default=8, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--floatX", default='float32')
    parser.add_argument('--sample_iterations', type=int, default=1000, help='sample mols and compute metrics')
    parser.add_argument("--log_weight_score", action='store_true', default=False)

    # objectives
    parser.add_argument("--objectives", type=str,
                        default='gsk3b,jnk3,qed,sa')  
    parser.add_argument("--acq_fn", default='UCB', type=str)
    parser.add_argument("--beta", default=0.1, type=float) 
    parser.add_argument("--scalar", default='WeightedSum', type=str)
    parser.add_argument("--alpha", default=1., type=float,
                        help='dirichlet distribution')
    parser.add_argument("--alpha_vector", default='1,1,1,1', type=str) 
    
    # Proxy
    parser.add_argument("--proxy_normalize", action='store_true',
                        default=False, help='normalize Y')
    parser.add_argument("--proxy_num_iterations", default=10000, type=int)
    parser.add_argument("--proxy_learning_rate", default=2.5e-4,
                        help="Learning rate", type=float)
    parser.add_argument("--proxy_mbsize", default=64,
                        help="Minibatch size", type=int)
    parser.add_argument("--proxy_early_stop_tol", default=10, type=int)
    parser.add_argument("--proxy_repr_type", default='atom_graph')
    parser.add_argument("--proxy_model_version", default='v2')
    parser.add_argument("--proxy_num_conv_steps",
                        default=12, type=int) 
    parser.add_argument("--proxy_nemb", default=64, help="#hidden", type=int)
    parser.add_argument("--proxy_weight_decay", default=1e-6,
                        help="Weight Decay in Proxy", type=float)
    parser.add_argument("--proxy_uncertainty", default="evidential", type=str) # deep ensemble and GP
    parser.add_argument("--proxy_dropout", default=0.1,
                        help="MC Dropout in Proxy", type=float)
    parser.add_argument("--proxy_num_dropout_samples", default=5, type=int)
    parser.add_argument("--evidential_lam", default=0.1, type=float)    
    parser.add_argument(
        "--fp_radius", type=int, default=2, help="Morgan fingerprint radius."
    )
    parser.add_argument(
        "--fp_nbits", type=int, default=1024, help="Morgan fingerprint nBits."
    )

    # GFlowNet
    parser.add_argument("--min_blocks", default=2, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--num_iterations", default=5000, type=int) 
    parser.add_argument("--criterion", default="FM", type=str)
    parser.add_argument("--learning_rate", default=5e-4,
                        help="Learning rate", type=float)
    parser.add_argument("--Z_learning_rate", default=5e-3,
                        help="Learning rate", type=float)
    parser.add_argument("--clip_grad", default=0, type=float)
    parser.add_argument("--trajectories_mbsize", default=8, type=int)
    parser.add_argument("--offline_mbsize", default=8, type=int)
    parser.add_argument("--hindsight_prob", default=0.2, type=float)
    parser.add_argument("--hindsight_buffer_mbsize", default=8, type=int)
    parser.add_argument("--hindsight_trajectories_mbsize", default=8, type=int)
    parser.add_argument("--reward_min", default=1e-2, type=float)
    parser.add_argument("--reward_norm", default=1, type=float)
    parser.add_argument("--reward_exp", default=8, type=float)
    parser.add_argument("--reward_exp_ramping", default=0, type=float)
    parser.add_argument("--logit_clipping", default=0., type=float)

    # Hyperparameters for TB
    parser.add_argument("--partition_init", default=1, type=float)

    # Hyperparameters for FM
    parser.add_argument("--log_reg_c", default=(0.1/8)**4, type=float)
    parser.add_argument("--balanced_loss", default=True)
    parser.add_argument("--leaf_coef", default=10, type=float)
    # Architecture
    parser.add_argument("--repr_type", default='block_graph')
    parser.add_argument("--model_version", default='v4')
    parser.add_argument("--num_conv_steps", default=10, type=int)
    parser.add_argument("--nemb", default=256, help="#hidden", type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--random_action_prob", default=0.05, type=float)
    parser.add_argument("--bootstrap_tau", default=0, type=float)
    parser.add_argument("--condition_type", type=str, default='HN')
    parser.add_argument("--ray_hidden_dim", default=100, type=int)

    return parser.parse_args()
    

class BoRolloutWorker(RolloutWorker):
    def __init__(self, args, bpath, proxy, device):
        super(BoRolloutWorker, self).__init__(args, bpath, proxy, device)
        self.hindsight_prob = args.hindsight_prob
        self.hindsight_mols = defaultdict(list)
        self.hindsight_smiles = defaultdict(list)
        self.replay_threshold = 0.9

    def _get(self, i, dset, weights=None):
        # Sample trajectories by walking backwards from the molecules in our dataset
        # Handle possible multithreading issues when independent threads
        # add/substract from dset:
        m = dset[i]

        if not isinstance(m, BlockMoleculeDataExtended):
            m = m[-1]

        r, raw_r = self._get_reward(m, weights)
        done = 1
        samples = []
        # a sample is a tuple (parents(s), parent actions, reward(s), s, done)
        # an action is (blockidx, stemidx) or (-1, x) for 'stop'
        # so we start with the stop action, unless the molecule is already
        # a "terminal" node (if it has no stems, no actions).
        if len(m.stems) and len(m.blocks) < self.max_blocks:
            samples.append(((m,), ((-1, 0),), weights, weights, r, m, done))
            r = done = 0
        while len(m.blocks):  # and go backwards
            if self.ignore_parents:
                parents = self.mdp.parents(m)
                parent, action = parents[self.train_rng.randint(len(parents))]
                samples.append(((parent,), (action,), weights, weights, r, m, done))
                r = done = 0
                m = parent
            else:
                parents, actions = zip(*self.mdp.parents(m))
                samples.append((parents, actions, weights.repeat(len(parents), 1), weights, r, m, done))
                r = done = 0
                m = parents[self.train_rng.randint(len(parents))]

        return samples[::-1]
    
    def _add_mol_to_replay(self, m):
        for i, weights in enumerate(self.test_weights):
            r, raw_r = self._get_reward(m, weights) 
            if len(self.hindsight_mols[i]) < self.max_hindsight_mols or raw_r[0] > self.hindsight_mols[i][0][0]:  
                if m.smiles not in self.hindsight_smiles[i]:
                    self.hindsight_mols[i].append((raw_r[0].item(), m.smiles, m))
                    self.hindsight_smiles[i].append(m.smiles)

            if len(self.hindsight_mols[i]) > self.max_hindsight_mols:
                self.hindsight_mols[i] = sorted(self.hindsight_mols[i], key=lambda x:(x[0]))[
                    max(int(0.05 * self.max_hindsight_mols), 1):]
                self.hindsight_smiles[i] = [x[1] for x in self.hindsight_mols[i]]
            
    def _add_mol_to_online(self, r, m, inflow):
        if self.replay_mode == 'online':
            r = r + self.train_rng.normal() * 0.01
            if len(self.online_mols) < self.max_online_mols or r > self.online_mols[0][0]:
                self.online_mols.append((r, m))
            if len(self.online_mols) > self.max_online_mols:
                self.online_mols = sorted(self.online_mols)[
                    max(int(0.05 * self.max_online_mols), 1):]
        elif self.replay_mode == 'prioritized':
            self.online_mols.append((abs(inflow - np.log(r)), m))
            if len(self.online_mols) > self.max_online_mols * 1.1:
                self.online_mols = self.online_mols[-self.max_online_mols:]
        
    def _get_reward(self, m, weights=None):
        rdmol = m.mol
        if rdmol is None:
            return self.reward_min
        
        # get reward from proxy
        raw_reward, score = self.proxy(m, weights)
        raw_reward = raw_reward.clip(self.reward_min)
            
        reward = self.l2r(raw_reward)

        return reward, (raw_reward, score)

    def execute_train_episode_batch(self, generator, dataset=None, Y_bounds=None, use_rand_policy=True):
        if self.train_rng.uniform() < self.hindsight_prob:
            idx = self.train_rng.randint(self.test_weights.shape[0])
            weights = self.test_weights[idx].unsqueeze(0)
            
            samples = sum((self.rollout(generator, use_rand_policy, weights)
                    for i in range(self.args.hindsight_trajectories_mbsize)), [])
            
            if self.args.hindsight_buffer_mbsize > 0:
                buffer = deepcopy(self.hindsight_mols[idx])
                reward = np.array([x[0] for x in buffer])
                prob = reward / sum(reward)
                eidx = np.random.choice(list(range(len(buffer))), self.args.hindsight_buffer_mbsize, replace=False, p=prob)
                offline_samples = sum((self._get(i, buffer, weights) 
                                    for i in eidx), [])
                samples += offline_samples    
        else:
            weights = Dirichlet(torch.tensor(self.args.alpha_vector)*self.args.alpha).sample_n(1).to(self.args.device) #* sample weights per batch, seem better
            samples = sum((self.rollout(generator, use_rand_policy, weights, replay=True)
                           for i in range(self.args.trajectories_mbsize)), [])
                
            # offline sampling from dataset
            if self.args.offline_mbsize > 0 and dataset is not None:
                # use the oracle reward
                scores = torch.tensor(pd.DataFrame.from_dict(dataset.scores).values, dtype=torch.float32).to(args.device)
                if Y_bounds is not None:
                    scores = normalize(scores, Y_bounds)
                    
                reward = torch.matmul(scores, weights.reshape(-1, 1))
                prob = (reward / sum(reward)).squeeze(1).cpu().numpy()
                eidx = np.random.choice(list(range(len(dataset.all_mols))), self.args.offline_mbsize, replace=False, p=prob)
                offline_samples = sum((self._get(i, dataset.all_mols, weights) 
                                    for i in eidx), [])
                samples += offline_samples

        return zip(*samples)
    
    def initialize_hindsight_mols(self, dataset):
        for m in dataset.all_mols:
            for i, weights in enumerate(self.test_weights):
                r, raw_r = self._get_reward(m, weights)
                self.hindsight_mols[i].append((raw_r[0].item(), m.smiles, m))
        for i, weights in enumerate(self.test_weights):       
            self.hindsight_mols[i] = sorted(self.hindsight_mols[i], key=lambda x:(x[0]))
            self.hindsight_smiles[i] = [x[1] for x in self.hindsight_mols[i]]

def train_generative_model(args, generator, bpath, proxy, oracle, dataset, test_weights, 
                           round_idx, do_save=False):
    print("Training generator...")
    os.makedirs(os.path.join(args.log_dir, f'round_{round_idx}'), exist_ok=True)
    device = args.device
    rollout_worker = BoRolloutWorker(args, bpath, proxy, device)
    rollout_worker.test_weights = torch.tensor(test_weights).to(device)
    rollout_worker.initialize_hindsight_mols(dataset)
     
    Y_bounds = torch.stack([proxy.partitioning.Y.min(dim=-2).values, proxy.partitioning.Y.max(dim=-2).values])

    def save_stuff(round_idx, iter):
        torch.save(generator.state_dict(), os.path.join(
            args.log_dir, 'round_{}/{}_generator_checkpoint.pth'.format(round_idx, iter)))
        pickle.dump(rollout_worker.sampled_mols,
                    gzip.open(f'{args.log_dir}/sampled_mols.pkl.gz', 'wb'))

    multi_thread = not args.debug
    if multi_thread:
        sampler = rollout_worker.start_samplers(generator, 8, dataset)

    def stop_everything():
        print('joining')
        rollout_worker.stop_samplers_and_join()

    last_losses = []
    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_last_check = time.time()

    for i in range(args.num_iterations + 1):
        if multi_thread:
            r = sampler()
            for thread in rollout_worker.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            p, pb, a, pw, w, r, s, d, mols = r
        else:
            p, pb, a, pw, w, r, s, d, mols = rollout_worker.sample2batch(
                rollout_worker.execute_train_episode_batch(generator, dataset, Y_bounds, use_rand_policy=True))

        loss = generator.train_step(p, pb, a, pw, w, r, s, d, mols, i)
        last_losses.append(loss)

        if not i % 100:
            train_loss = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            train_losses.append(train_loss)
            args.logger.add_scalar(
                'Loss/round{}/train'.format(round_idx), train_loss[0], use_context=False)
            print('Iter {}: Loss {}, Time {}'.format(
                i, train_loss, round(time.time() - time_last_check, 3)))
            time_last_check = time.time()
            last_losses = []

            if not i % args.sample_iterations and i != 0:  
                volume, volume_oracle, reward_weight, reward_mean, test_loss, diversity = sample_batch(
                    args, generator, rollout_worker, oracle, proxy, Y_bounds, compute_multi_objective_metric=False)
                args.logger.add_scalar(
                    'round{}/Top-100-sampled/volumes'.format(round_idx), volume, use_context=False)
                args.logger.add_scalar(
                    'round{}/Top-100-sampled/volumes_oracle'.format(round_idx), volume_oracle, use_context=False)
                args.logger.add_scalars(
                    'round{}/Top-100-sampled/reward_weight'.format(round_idx), reward_weight, use_context=False) 
                args.logger.add_scalar(
                    'round{}/Top-100-sampled/reward_mean'.format(round_idx), reward_mean, use_context=False) # reward_mean is a dict, the keys are test_weights
                args.logger.add_scalar(
                    'round{}/Top-100-sampled/test_loss'.format(round_idx), test_loss, use_context=False)
                args.logger.add_scalar(
                    'round{}/Top-100-sampled/dists'.format(round_idx), diversity, use_context=False)
                
                if do_save:
                    save_stuff(round_idx, i)
                        
    stop_everything()
    if do_save:
        save_stuff(round_idx, i)
    
    checkpoint_path = os.path.join(args.log_dir, f'round_{round_idx}/{i}_generator_checkpoint.pth')
    generator.load_state_dict(torch.load(checkpoint_path))
    
    return rollout_worker, {'train_losses': train_losses,
                            'test_losses': test_losses,
                            'test_infos': test_infos,
                            'train_infos': train_infos}

def sample_batch(args, generator, rollout_worker, oracle=None, proxy=None, ref_mols=None, Y_bounds=None, compute_multi_objective_metric=False):
    score_succ = {'gsk3b': 0.5, 'jnk3': 0.5, 'drd2': 0.5, 
                  'chemprop_sars': 0.5, 'chemprop_hiv': 0.5, "seh": 0.5,
                  'qed': 0.6, 'sa': 0.67}
    if Y_bounds is None:
        Y_bounds = torch.stack([proxy.partitioning.Y.min(
                    dim=-2).values, proxy.partitioning.Y.max(dim=-2).values])
    
    time_start = time.time()
    print(f"Sampling molecules...")
    raw_rewards = []
    raw_rewards_weight = {}
    means = []
    picked_mols = []
    smis = []

    for i, weights in enumerate(rollout_worker.test_weights):
        sampled_mols = []
        sampled_raw_rewards = []
        sampled_means = []
        sampled_smis = []
        while len(sampled_mols) < args.num_samples: 
            rollout_worker.rollout(generator, use_rand_policy=False, weights=torch.tensor(weights).unsqueeze(0).to(args.device))
            (raw_r, _, m, trajectory_stats, inflow) = rollout_worker.sampled_mols[-1]
            sampled_mols.append(m)
            sampled_raw_rewards.append(raw_r[0].item())
            sampled_means.append(raw_r[1])
            sampled_smis.append(m.smiles)
                
        idx_pick = np.argsort(sampled_raw_rewards)[::-1][:int(args.num_samples/len(rollout_worker.test_weights))]
        picked_mols.extend(np.array(sampled_mols)[idx_pick].tolist())
        means.extend(np.array(sampled_means)[idx_pick].tolist())
        smis.extend(np.array(sampled_smis)[idx_pick].tolist())
        raw_rewards.extend(np.array(sampled_raw_rewards)[idx_pick].tolist())
        raw_rewards_weight[str(weights.cpu())] = np.array(sampled_raw_rewards)[idx_pick].mean()
        
    raw_rewards_mean = np.mean(list(raw_rewards_weight.values()))
    assert len(picked_mols) == args.num_samples

    top_means = torch.tensor(means)
    scores_dict = oracle.batch_get_scores(picked_mols)
    scores = torch.tensor(pd.DataFrame.from_dict(scores_dict).values)
    test_loss = F.mse_loss(top_means, scores)

    hypervolume = Hypervolume(ref_point=torch.zeros(len(args.objectives)))
    volume = hypervolume.compute(top_means)
    volume_oracle = hypervolume.compute(scores)
    
    diversity = compute_diversity(picked_mols)
    
    batch_metrics = {'Hypervolume_reward': volume,
                     'Hypervolume_oracle': volume_oracle,
                     'Reward_mean': raw_rewards_mean,
                     'scores_max': pd.DataFrame.from_dict(scores_dict).max().to_dict(),
                     'scores_mean': pd.DataFrame.from_dict(scores_dict).mean().to_dict(),
                     'Test_loss': test_loss,
                     'Diversity': diversity}
    print(batch_metrics)
    print('Time: {}'.format(time.time()-time_start))
    
    if not compute_multi_objective_metric:
        return volume, volume_oracle, raw_rewards_weight, raw_rewards_mean, test_loss, diversity
        
    else:
        for i in range(len(picked_mols)):
            picked_mols[i].score = scores_dict[i]

        # success/diversity/novelty is computed among the top mols.
        success, positive_mols = compute_success(
            picked_mols, scores_dict, args.objectives, score_succ)
        succ_diversity = compute_diversity(positive_mols)
        if ref_mols:
            novelty = compute_novelty(positive_mols, ref_mols)
        else:
            novelty = 1.

        mo_metrics = {'success': success, 'novelty': novelty,
                      'succ_diversity': succ_diversity, }

        picked_smis = [(raw_rewards[i], picked_mols[i].score, smis[i])
                       for i in range(len(raw_rewards))]
        print(mo_metrics)
        return (picked_mols, scores_dict, picked_smis), batch_metrics, mo_metrics

def log_overall_metrics(args, dataset, batch_infos=None, MultiObjective_metrics=None):
    volume = dataset.compute_hypervolume()
    print("Hypervolume for {}: {}".format(args.logger.context, volume))
    args.logger.add_scalar('Metric/hypervolume', volume, use_context=False)
    args.logger.add_object('scores', dataset.scores)
    args.logger.add_object('smis', dataset.smis)

    if batch_infos:
        args.logger.add_scalar(
            'Metric/test_loss', batch_infos['Test_loss'], use_context=False)
        args.logger.add_object('collected_info', batch_infos)

    if MultiObjective_metrics:
        args.logger.add_scalars('Metric/MultiObjective',
                                MultiObjective_metrics, use_context=False)
        
def get_test_rays():
    if args.n_objectives == 3:
        n_partitions = 6
    elif args.n_objectives == 4:
        n_partitions = 7
    test_rays = get_reference_directions("das-dennis", args.n_objectives, n_partitions=n_partitions).astype(np.float32) 
    test_rays = test_rays[[(r > 0).all() for r in test_rays]]
    print(f"initialize {len(test_rays)} test rays") 
    return test_rays 

def main(args):
    set_random_seed(args.seed)
    args.logger.set_context('iter_0')
    bpath = "./data/blocks_105.json"
    dpath = "./data/docked_mols.h5"

    # Initialize oracle and dataset (for training surrogate function)
    oracle = Oracle(args)
    dataset = Dataset(args, bpath, oracle, args.device)
    dataset.load_h5(dpath, num_init_examples=args.num_init_examples)
    log_overall_metrics(args, dataset)    
    args.n_objectives = len(args.objectives)

    # Initialize surrogate function
    proxy = get_proxy(args, bpath, oracle)
    proxy.update(dataset, 0, reset=False)
            
    for i in range(1, args.num_outer_loop_iters+1):
        print(f"====== Starting round {i} ======")
        args.logger.set_context('iter_{}'.format(i))            
        test_weights = np.random.dirichlet(args.alpha_vector, 5*(2**(args.n_objectives-2))).astype(np.float32)

        if args.criterion == 'TB':
            generator = TBGFlowNet(args, bpath)
        elif args.criterion == 'FM':
            generator = FMGFlowNet(args, bpath)
        elif args.criterion == 'Reinforce':
            generator = MOReinforce(args, bpath)
        else:
            raise ValueError('Not implemented!')

        rollout_worker, training_metrics = train_generative_model(
            args, generator, bpath, proxy, oracle, dataset, test_weights, i, do_save=args.save)

        # sample molecule batch from generator and update dataset with oracle scores for sampled batch
        batch, batch_infos, MultiObjective_metrics = sample_batch(
            args, generator, rollout_worker, oracle, proxy, compute_multi_objective_metric=True)
        dataset.add_samples(batch)
        log_overall_metrics(args, dataset, batch_infos, MultiObjective_metrics)
        args.logger.save(os.path.join(args.log_dir, 'logged_data.pkl.gz'))

        # update proxy with new data
        if i != args.num_outer_loop_iters:
            proxy.update(dataset, i, reset=True)

if __name__ == '__main__':
    args = arg_parse()
    args.logger = get_logger(args)
    args.objectives = args.objectives.split(',')
    args.alpha_vector = args.alpha_vector.split(',')
    args.alpha_vector = [float(x) for x in args.alpha_vector]

    main(args)
