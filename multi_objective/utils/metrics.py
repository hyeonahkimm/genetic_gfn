from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from botorch.utils.multi_objective.hypervolume import Hypervolume
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy import stats
from scipy.special import kl_div
import networkx as nx
import itertools
import time
from tqdm import tqdm
from mol_mdp_ext import BlockMoleculeDataExtended


def compute_success(mols, scores, objectives, score_succ):
    print("Computing successful rate...")
    positive_mols = []
    success_dict = {k: 0. for k in objectives}

    for mol, score in zip(mols, scores):
        all_success = True
        for k, v in score.items():
            if v >= score_succ[k]:
                success_dict[k] += 1
            else:
                all_success = False
        if all_success:
            positive_mols.append(mol)

    success = 1.*len(positive_mols)/len(mols)

    return success, positive_mols


def compute_diversity(mols):
    # print("Computing diversity...")

    if len(mols) == 0:
        return 0

    sims = []
    if isinstance(mols[0], BlockMoleculeDataExtended):
        fps = [AllChem.GetMorganFingerprintAsBitVect(x.mol, 3, 2048) for x in mols]
    else:
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]
    for i in range(len(fps)):
        sims += DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])

    return 1 - np.mean(sims)


def compute_novelty(mols, ref_mols):
    print("Computing novelty...")
    positive_fps = [AllChem.GetMorganFingerprintAsBitVect(
        x.mol, 3, 2048) for x in mols]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(
        x, 3, 2048) for x in ref_mols]

    n_sim = 0.
    for i in range(len(positive_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(positive_fps[i], ref_fps)
        if max(sims) >= 0.4:
            n_sim += 1
    novelty = 1. - 1. * n_sim / (len(positive_fps)+1e-6)

    return novelty


def compute_uniformity(scores, weights):
    m = scores.shape[-1]
    weighted_score = weights * scores
    normalization = weighted_score / \
        np.sum(weighted_score, axis=1, keepdims=True)
    uniformity = 1 - kl_div(normalization, 1/m).sum(1)
    return uniformity.mean()


def evaluate(args, generator, rollout_worker, k):
    time_start = time.time()
    print(f"Sampling molecules and evaluating...")
    test_weights = rollout_worker.test_weights
    picked_mols = []
    all_scores = []
    # top_scores = []
    top_scores = defaultdict(list)
    mean_scores = []
    hypervolume = Hypervolume(ref_point=torch.zeros(len(args.objectives)))
    
    for weights in test_weights:
        sampled_mols = []
        rewards = []
        scores = []
        for i in range(args.num_samples):
            rollout_worker.rollout(
                generator, use_rand_policy=False, weights=weights.unsqueeze(0))
            (raw_r, _, m, _, _) = rollout_worker.sampled_mols[-1]
            sampled_mols.append(m)
            rewards.append(raw_r[0])
            scores.append(raw_r[1])

        idx_pick = np.argsort(rewards)[::-1][:k]  
        picked_mols += np.array(sampled_mols)[idx_pick].tolist()
        top_rewards = np.array(rewards)[idx_pick]
        mean_scores.append(np.array(scores).mean(0))
        
        picked_scores = np.array(scores)[idx_pick]
        weight_specific_volume = hypervolume.compute(torch.tensor(picked_scores))
        print(f'Hypervolume w.r.t test weights {weights}: {weight_specific_volume}')
        
        for K in [10, 100]:
            scores_np = np.array(scores)
            top_scores_weight = [scores_np[np.argsort(scores_np[:,i])[::-1][:K], i].mean() for i in range(len(args.objectives))]
            top_scores[K].append(top_scores_weight)
            print(f'Top {K} scores w.r.t test weights {weights}: {top_scores_weight}')
            
        all_scores += scores
        print('Top_rewards: {}'.format(top_rewards.mean()))  # Top-100 rewards
                
    volume = hypervolume.compute(torch.tensor(all_scores))
    diversity = compute_diversity(picked_mols)  # Top-100

    print('Hypervolume: {}, Diversity: {}, Time: {}'.format(
        volume, diversity, time.time()-time_start))

    return volume, diversity


def get_mol_path_graph(mol, mdp):
    # bpath = "data/blocks_PDB_105.json"
    # mdp = MolMDPExtended(bpath)
    # mdp.post_init(torch.device('cpu'), 'block_graph')
    # mdp.build_translation_table()
    # mdp.floatX = torch.float
    agraph = nx.DiGraph()
    agraph.add_node(0)
    ancestors = [mol]
    ancestor_graphs = []

    par = mdp.parents(mol)
    mstack = [i[0] for i in par]
    pstack = [[0, a] for i, a in par]
    while len(mstack):
        m = mstack.pop()  # pop = last item is default index
        p, pa = pstack.pop()
        match = False
        mgraph = mdp.get_nx_graph(m)
        for ai, a in enumerate(ancestor_graphs):
            if mdp.graphs_are_isomorphic(mgraph, a):
                agraph.add_edge(p, ai+1, action=pa)
                match = True
                break
        if not match:
            # I assume the original molecule = 0, 1st ancestor = 1st parent = 1
            agraph.add_edge(p, len(ancestors), action=pa)
            # so now len(ancestors) will be 2 --> and the next edge will be to the ancestor labelled 2
            ancestors.append(m)
            ancestor_graphs.append(mgraph)
            if len(m.blocks):
                par = mdp.parents(m)
                mstack += [i[0] for i in par]
                pstack += [(len(ancestors)-1, i[1]) for i in par]

    for u, v in agraph.edges:
        c = mdp.add_block_to(ancestors[v], *agraph.edges[(u, v)]['action'])
        geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                        mdp.get_nx_graph(ancestors[u], true_block=True))
        if not geq:  # try to fix the action
            block, stem = agraph.edges[(u, v)]['action']
            for i in range(len(ancestors[v].stems)):
                c = mdp.add_block_to(ancestors[v], block, i)
                geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                                mdp.get_nx_graph(ancestors[u], true_block=True))
                if geq:
                    agraph.edges[(u, v)]['action'] = (block, i)
                    break
        if not geq:
            raise ValueError('could not fix action')
    for u in agraph.nodes:
        agraph.nodes[u]['mol'] = ancestors[u]
    return agraph


def compute_correlation(args, model, rollout_worker, test_mols):

    mdp = rollout_worker.mdp
    device = args.device
    def tf(x): return torch.tensor(x, device=device).to(torch.float)
    def tint(x): return torch.tensor(x, device=device).long()

    # test_mols = pickle.load(gzip.open('data/some_mols_U_1k.pkl.gz'))
    logsoftmax = nn.LogSoftmax(0)
    corrs = []
    numblocks = []

    start_time = time.time()
    if args.n_objectives == 3:
        test_weights = rollout_worker.test_weights[::2]
    elif args.n_objectives == 4:
        test_weights = rollout_worker.test_weights[1:-2:4]
    else:
        test_weights = rollout_worker.test_weights
                
    for weights in test_weights:
        print("Computing correlation w.r.t test weights {}".format(weights))
        weights = torch.tensor(weights).to(args.device)
        logp = []
        rewards = []
        for m in tqdm(test_mols):
            try:
                agraph = get_mol_path_graph(m, mdp)
            except:
                continue
            # rewards.append(np.log(moli[0][0]))
            reward = rollout_worker._get_reward(m, weights)[0].item()
            rewards.append(np.log(reward))
            s = mdp.mols2batch([mdp.mol2repr(agraph.nodes[i]['mol'])
                                for i in agraph.nodes])
            numblocks.append(len(m.blocks))
            with torch.no_grad():
                # get the mols_out_s for ALL molecules not just the end one.
                if args.condition_type == 'Hyper_scorepred':
                    stem_out_s, mol_out_s, _ = model(
                        s, weights.repeat(s.num_graphs, 1))
                else:
                    stem_out_s, mol_out_s = model(
                        s, weights.repeat(s.num_graphs, 1))
            per_mol_out = []
            # Compute pi(a|s)
            for j in range(len(agraph.nodes)):
                a, b = s._slice_dict['stems'][j:j+2]

                stop_allowed = len(
                    agraph.nodes[j]['mol'].blocks) >= args.min_blocks
                mp = logsoftmax(torch.cat([
                    stem_out_s[a:b].reshape(-1),
                    # If num_blocks < min_blocks, the model is not allowed to stop
                    mol_out_s[j, :1] if stop_allowed else tf([-1000])]))
                per_mol_out.append(
                    (mp[:-1].reshape((-1, stem_out_s.shape[1])), mp[-1]))

            # When the model reaches 8 blocks, it is stopped automatically. If instead it stops before
            # that, we need to take into account the STOP action's logprob
            if len(m.blocks) < 8:
                if args.condition_type == 'Hyper_scorepred':
                    stem_out_last, mol_out_last, _ = model(
                        mdp.mols2batch([mdp.mol2repr(m)]), weights.unsqueeze(0))
                else:
                    stem_out_last, mol_out_last = model(
                        mdp.mols2batch([mdp.mol2repr(m)]), weights.unsqueeze(0))                   
                mplast = logsoftmax(
                    torch.cat([stem_out_last.reshape(-1), mol_out_last[0, :1]]))
                MSTOP = mplast[-1]

            # assign logprob to edges
            for u, v in agraph.edges:
                a = agraph.edges[u, v]['action']
                if a[0] == -1:
                    agraph.edges[u, v]['logprob'] = per_mol_out[v][1]
                else:
                    agraph.edges[u,
                                 v]['logprob'] = per_mol_out[v][0][a[1], a[0]]

            # propagate logprobs through the graph
            for n in list(nx.topological_sort(agraph))[::-1]:
                for c in agraph.predecessors(n):
                    if len(m.blocks) < 8 and c == 0:
                        agraph.nodes[c]['logprob'] = torch.logaddexp(
                            agraph.nodes[c].get('logprob', tf(-1000)),
                            agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob', 0) + MSTOP)
                    else:
                        agraph.nodes[c]['logprob'] = torch.logaddexp(
                            agraph.nodes[c].get('logprob', tf(-1000)),
                            agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob', 0))

            # add the first item
            # logp.append((moli, agraph.nodes[n]['logprob'].item()))
            logp.append(agraph.nodes[n]['logprob'].item())
        corrs.append(stats.spearmanr(rewards, logp).correlation)

    print('Spearmanr: {}, mean: {}, Time: {}'.format(corrs, np.mean(corrs), time.time()-start_time))
    return corrs


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    weights = np.c_[x, y]
    normalized_weights = weights/weights.sum(1, keepdims=True)

    return normalized_weights.astype(np.float32)
