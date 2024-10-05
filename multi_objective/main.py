from curses import raw
import os
from dataset import Dataset
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from oracle.oracle import Oracle
# from proxy import get_proxy
from generator import TBGFlowNet, FMGFlowNet
from utils.metrics import circle_points, compute_success, compute_diversity, compute_novelty, evaluate, compute_correlation
from utils.utils import set_random_seed
from utils.logging import get_logger
from datetime import datetime
import argparse
import json
import time
import threading
import pdb
import pickle
import gzip
import warnings
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch
from torch.distributions.dirichlet import Dirichlet
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
warnings.filterwarnings('ignore')


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("--run", default=0, help="run", type=int)
    parser.add_argument('--save', action='store_true',
                        default=False, help='Save model.')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode, no multi thread')
    parser.add_argument("--enable_tensorboard",
                        action='store_true', default=False)
    parser.add_argument("--log_dir", default='runs/synthetic')
    parser.add_argument("--include_nblocks", default=False)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--floatX", default='float32')
    parser.add_argument('--sample_iterations', type=int, default=1000, help='sample mols and compute metrics')

    # objectives
    parser.add_argument("--objectives", type=str, default='gsk3b,jnk3')
    parser.add_argument("--scalar", default='WeightedSum', type=str) #TODO: other scalars
    parser.add_argument("--alpha", default=1., type=float,
                        help='dirichlet distribution')
    parser.add_argument("--alpha_vector", default='1,1', type=str) 

    # GFlowNet
    parser.add_argument("--min_blocks", default=2, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--num_iterations", default=30000, type=int)  # 30k
    parser.add_argument("--criterion", default="FM", type=str)
    parser.add_argument("--learning_rate", default=5e-4,
                        help="Learning rate", type=float)
    parser.add_argument("--Z_learning_rate", default=5e-3,
                        help="Learning rate", type=float)
    parser.add_argument("--clip_grad", default=0, type=float)
    parser.add_argument("--trajectories_mbsize", default=16, type=int)
    parser.add_argument("--offline_mbsize", default=0, type=int)
    parser.add_argument("--hindsight_mbsize", default=0, type=int)
    parser.add_argument("--reward_min", default=1e-2, type=float)
    parser.add_argument("--reward_norm", default=0.8, type=float)
    parser.add_argument("--reward_exp", default=6, type=float)
    parser.add_argument("--reward_exp_ramping", default=0, type=float)

    # Hyperparameters for TB
    parser.add_argument("--partition_init", default=30, type=float)

    # Hyperparameters for FM
    parser.add_argument("--log_reg_c", default=(0.1/8)
                        ** 4, type=float)  # (0.1/8)**8  
    parser.add_argument("--balanced_loss", default=True)
    parser.add_argument("--leaf_coef", default=10, type=float)
    # Architecture
    parser.add_argument("--repr_type", default='block_graph')
    parser.add_argument("--model_version", default='v4')
    parser.add_argument("--condition_type", default='HN', type=str)  # 'HN', 'FiLM', 'concat'
    parser.add_argument("--num_conv_steps", default=10, type=int)
    parser.add_argument("--nemb", default=256, help="#hidden", type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--random_action_prob", default=0.05, type=float)
    parser.add_argument("--bootstrap_tau", default=0, type=float)
    parser.add_argument("--ray_hidden_dim", default=100, type=int)
    parser.add_argument("--logit_clipping", default=0., type=float)

    return parser.parse_args()


class RolloutWorker:
    def __init__(self, args, bpath, proxy, device):
        self.args = args
        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(int(time.time()))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.repr_type,
                           include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        if args.floatX == 'float64':
            self.mdp.floatX = self.floatX = torch.double
        else:
            self.mdp.floatX = self.floatX = torch.float
        self.proxy = proxy
        self._device = device
        self.seen_molecules = set()
        self.stop_event = threading.Event()
        #######
        # This is the "result", here a list of (reward, BlockMolDataExt, info...) tuples
        self.sampled_mols = []
        self.online_mols = []
        self.hindsight_mols = []
        self.max_online_mols = 1000
        self.max_hindsight_mols = 1000

        self.min_blocks = args.min_blocks
        self.max_blocks = args.max_blocks
        self.mdp._cue_max_blocks = self.max_blocks
        self.reward_exp = args.reward_exp
        self.reward_min = args.reward_min
        self.reward_norm = args.reward_norm
        self.reward_exp_ramping = args.reward_exp_ramping
        self.random_action_prob = args.random_action_prob

        # If True this basically implements Buesing et al's TreeSample Q,
        # samples uniformly from it though, no MTCS involved
        if args.criterion == 'TB' or args.criterion == "Reinforce":
            self.ignore_parents = True
        elif args.criterion == 'FM':
            self.ignore_parents = False

    def rollout(self, generator, use_rand_policy=True, weights=None, replay=False):
        weights = Dirichlet(torch.ones(len(self.args.objectives))*self.args.alpha).sample_n(1).to(
            self.args.device) if weights is None else weights

        m = BlockMoleculeDataExtended()
        samples = []
        max_blocks = self.max_blocks
        trajectory_stats = []
        for t in range(max_blocks):
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            s_o, m_o = generator(s, vec_data=weights, do_stems=True)
            # fix from run 330 onwards
            if t < self.min_blocks:
                m_o = m_o*0 - 1000  # prevent assigning prob to stop
                # when we can't stop
            ##
            logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            cat = torch.distributions.Categorical(
                logits=logits) 
            action = cat.sample().item()

            if use_rand_policy and self.random_action_prob > 0: # just for training
                if self.train_rng.uniform() < self.random_action_prob:
                    action = self.train_rng.randint(
                        int(t < self.min_blocks), logits.shape[0])

            q = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            trajectory_stats.append(
                (q[action].item(), action, torch.logsumexp(q, 0).item()))

            if t >= self.min_blocks and action == 0:
                r, raw_r = self._get_reward(m, weights)  # r: reward, raw_r: scores for the objectives
                samples.append(((m,), ((-1, 0),), weights, weights, r, m, 1))
                break
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks,
                          action // self.mdp.num_blocks)
                m_old = m
                m = self.mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r, raw_r = self._get_reward(m, weights)
                    if self.ignore_parents:
                        samples.append(
                            ((m_old,), (action,), weights, weights, r, m, 1))
                    else:
                        parents, actions = zip(*self.mdp.parents(m))
                        samples.append((parents, actions, weights.repeat(
                            len(parents), 1), weights, r, m, 1))
                    break
                else:
                    if self.ignore_parents:
                        samples.append(
                            ((m_old,), (action,), weights, weights, 0, m, 0))
                    else:
                        parents, actions = zip(*self.mdp.parents(m))
                        samples.append(
                            (parents, actions, weights.repeat(len(parents), 1), weights, 0, m, 0))

        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in samples[-1][0]])
        qp = generator(p, weights.repeat(p.num_graphs, 1))
        qsa_p = generator.model.index_output_by_action(
            p, qp[0], qp[1][:, 0],
            torch.tensor(samples[-1][1], device=self._device).long())
        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        self.sampled_mols.append(
            ([i.cpu().numpy() for i in raw_r], weights.cpu().numpy(), m, trajectory_stats, inflow))

        if replay and self.args.hindsight_prob > 0.0:
            self._add_mol_to_replay(m)

        return samples

    def _get_reward(self, m, weights=None):
        rdmol = m.mol
        if rdmol is None:
            return self.reward_min
        
        # get scores from oracle
        score = self.proxy.get_score([m])
        score = torch.tensor(list(score.values())).to(self.args.device)
        
        if self.args.scalar == 'WeightedSum':
            raw_reward = (weights*score).sum()
        
        elif self.args.scalar == 'Tchebycheff':
            raw_reward = (weights*score).min() + 0.1 * (weights*score).sum()
        
        reward = self.l2r(raw_reward.clip(self.reward_min))
        return reward, (raw_reward, score)

    def execute_train_episode_batch(self, generator, dataset=None, use_rand_policy=True):
        if self.args.condition_type is None:
            weights = self.test_weights  # train specific model
        else:
            weights = Dirichlet(torch.tensor(self.args.alpha_vector)*self.args.alpha).sample_n(1).to(self.args.device) #* sample weights per batch, seem better
        samples = sum((self.rollout(generator, use_rand_policy, weights)
                      for i in range(self.args.trajectories_mbsize)), [])

        return zip(*samples)

    def sample2batch(self, mb):
        p, a, p_weights, weights, r, s, d, *o = mb
        mols = (p, s)
        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i, p in enumerate(p)], []),
                               device=self._device).long()
        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need
        # p_batch
        p = self.mdp.mols2batch(list(map(self.mdp.mol2repr, sum(p, ()))))
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        # Concatenate all the actions (one per parent per sample)
        a = torch.tensor(sum(a, ()), device=self._device).long()
        # rewards and dones
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        # weights
        p_w = torch.cat(p_weights, 0)
        w = torch.cat(weights, 0)
        return (p, p_batch, a, p_w, w, r, s, d, mols, *o)

    def l2r(self, raw_reward, t=0):
        if self.reward_exp_ramping > 0:
            reward_exp = 1 + (self.reward_exp - 1) * \
                (1 - 1/(1 + t / self.reward_exp_ramping))
            # when t=0, exp = 1; t->∞, exp = self.reward_exp
        else:
            reward_exp = self.reward_exp

        reward = (raw_reward/self.reward_norm)**reward_exp

        return reward

    def start_samplers(self, generator, n, dataset):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n

        def f(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(
                        self.execute_train_episode_batch(generator, dataset, use_rand_policy=True))
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()

        self.sampler_threads = [threading.Thread(
            target=f, args=(i,)) for i in range(n)]
        [setattr(i, 'failed', False) for i in self.sampler_threads]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]

        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
            while any([i.is_alive() for i in self.sampler_threads]):
                [i.set() for i in self.resume_events]
                [i.join(0.05) for i in self.sampler_threads]


def train_generative_model_with_oracle(args, generator, bpath, oracle, test_weights, dataset=None, do_save=False):
    print("Training generator...")
    device = args.device
    rollout_worker = RolloutWorker(args, bpath, oracle, device)
    if args.condition_type is None:
        rollout_worker.test_weights = torch.tensor(test_weights).to(device)[args.run :args.run+1]
    else:
        rollout_worker.test_weights = torch.tensor(test_weights).to(device)
    rollout_worker.test_mols = pickle.load(gzip.open('./data/test_mols_6062.pkl.gz', 'rb'))

    def save_stuff(iter):
        torch.save(generator.state_dict(), os.path.join(
            args.log_dir, '{}_generator_checkpoint.pth'.format(iter)))
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
    best_hv = 0
    best_corr = 0
    time_last_check = time.time()

    for i in range(args.num_iterations + 1):
        rollout_worker.reward_exp = 1 + (args.reward_exp-1) * (1-1/(1+i/20))
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
                rollout_worker.execute_train_episode_batch(generator, dataset, use_rand_policy=True))

        loss = generator.train_step(p, pb, a, pw, w, r, s, d, mols, i)
        last_losses.append(loss)

        if not i % 100:
            train_loss = [np.round(np.mean(loss), 3)
                          for loss in zip(*last_losses)]
            train_losses.append(train_loss)
            args.logger.add_scalar(
                'Loss/train', train_loss[0], use_context=False)
            print('Iter {}: Loss {}, Time {}'.format(
                i, train_loss, round(time.time() - time_last_check, 3)))
            time_last_check = time.time()
            last_losses = []

            if not i % args.sample_iterations and i != 0:
                volume, diversity = evaluate(args, generator, rollout_worker, 100)
                corrs = compute_correlation(args, generator, rollout_worker, rollout_worker.test_mols)

                args.logger.add_scalar(
                    'Top-100-sampled/volumes', volume, use_context=False)
                args.logger.add_scalar(
                    'Top-100-sampled/dists', diversity, use_context=False)
                args.logger.add_scalar(
                    'Top-100-sampled/corr', np.mean(corrs), use_context=False)
                if do_save:
                    save_stuff(i)

                if volume > best_hv:
                    best_hv = volume
                    if do_save:
                        save_stuff('volume')

                if np.mean(corrs) > best_corr:
                    best_corr = np.mean(corrs)
                    if do_save:
                        save_stuff('corr')

    stop_everything()
    if do_save:
        save_stuff(i)
    return rollout_worker, {'train_losses': train_losses,
                            'test_losses': test_losses,
                            'test_infos': test_infos,
                            'train_infos': train_infos}
        
def get_test_mols(args, mdp, num):
    samples = []
    fps = []
    early_stops = []
    while len(samples) < num:
        if len(samples) % 5000 == 0:
            print(f'{len(samples)}/{num} mols have been sampled')
        m = BlockMoleculeDataExtended()
        min_blocks = args.min_blocks
        max_blocks = args.max_blocks
        early_stop_at = np.random.randint(min_blocks, max_blocks + 1)
        early_stops.append(early_stop_at)
        for t in range(max_blocks):
            if t == 0:
                length = mdp.num_blocks+1
            else:
                length = len(m.stems)*mdp.num_blocks+1

            action = np.random.randint(1, length)

            if t == early_stop_at:
                action = 0

            if t >= min_blocks and action == 0:
                fp = AllChem.GetMorganFingerprintAsBitVect(m.mol, 3, 2048)
                if len(samples)==0:
                    samples.append(m)
                    fps.append(fp)
                else:
                    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
                    if max(sims) < 0.7:
                        samples.append(m)
                        fps.append(fp)
                break
            else:
                action = max(0, action-1)
                action = (action % mdp.num_blocks, action // mdp.num_blocks)
                #print('..', action)
                m = mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    fp = AllChem.GetMorganFingerprintAsBitVect(m.mol, 3, 2048)
                    if len(samples)==0:
                        samples.append(m)
                        fps.append(fp)
                    else:
                        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
                        if max(sims) < 0.7:
                            samples.append(m)
                            fps.append(fp)
                    break
                
    return samples

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

    # Initialization: oracle and dataset
    oracle = Oracle(args)
    args.n_objectives = len(args.objectives)
    if args.n_objectives == 2:
        test_weights = circle_points(K=5, min_angle=0.1, max_angle=np.pi/2-0.1)
    else:
        test_weights = get_test_rays()

    if args.criterion == 'TB':
        generator = TBGFlowNet(args, bpath)
    elif args.criterion == 'FM':
        generator = FMGFlowNet(args, bpath)
            
    else:
        raise ValueError('Not implemented!')

    rollout_worker, training_metrics = train_generative_model_with_oracle(
        args, generator, bpath, oracle, test_weights, do_save=args.save)

    args.logger.save(os.path.join(args.log_dir, 'logged_data.pkl.gz'))

if __name__ == '__main__':
    args = arg_parse()
    args.logger = get_logger(args)
    args.objectives = args.objectives.split(',')
    args.alpha_vector = args.alpha_vector.split(',')
    args.alpha_vector = [float(x) for x in args.alpha_vector]

    main(args)
