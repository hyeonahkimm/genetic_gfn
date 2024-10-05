from __future__ import print_function

import torch
import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import genetic_gfn.genetic_operator.crossover as co
import genetic_gfn.genetic_operator.mutate as mu

import gc


MINIMUM = 1e-10


def select_pop(population_mol: List[str], population_scores, population_size: int, rank_coefficient=0.01):
    scores_np = np.array(population_scores)
    ranks = np.argsort(np.argsort(-1 * scores_np))
    weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)
    
    indices = list(torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=population_size, replacement=False
        ))
    mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
    mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]
    
    return mating_pool, mating_pool_score


def make_mating_pool(population_mol: List[Mol], population_scores, population_size: int, rank_coefficient=0.01, replacement=True):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs 
    if rank_coefficient > 0:
        scores_np = np.array(population_scores)
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)
        
        indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=population_size, replacement=True
            ))
        mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]
        # print(mating_pool)
    else:
        population_scores = [s + MINIMUM for s in population_scores]
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
        # mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
        indices = np.random.choice(np.arange(len(population_mol)), p=population_probs, size=population_size, replace=True)
        mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]

    return mating_pool, mating_pool_score


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


class GeneticOperatorHandler:
    def __init__(self, mutation_rate: float=0.067, population_size=200):
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    # def get_final_population(self, mating_pool, rank_coefficient=0.):
    #     new_mating_pool, new_mating_scores, _, _ = make_mating_pool(mating_pool[0], mating_pool[1], self.population_size, rank_coefficient)
    #     return (new_mating_pool, new_mating_scores)

    def query(self, query_size, mating_pool, pool, rank_coefficient=0.01, mutation_rate=None):
        # print(mating_pool)
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        # import pdb; pdb.set_trace()
        # population_smi, population_scores = select_pop(mating_pool[0], mating_pool[1], self.population_size, rank_coefficient=rank_coefficient)
        # population_mol = [Chem.MolFromSmiles(s) for s in population_smi]
        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        cross_mating_pool, cross_mating_scores = make_mating_pool(population_mol, population_scores, query_size, rank_coefficient)

        offspring_mol = pool(delayed(reproduce)(cross_mating_pool, mutation_rate) for _ in range(query_size))
        new_mating_pool = cross_mating_pool
        new_mating_scores = cross_mating_scores

        smis, n_atoms = [], []
        for m in offspring_mol:
            try:
                # smis.append(Chem.MolToSmiles(m))
                smi = Chem.MolToSmiles(m)
                if smi not in smis:  # unique
                    smis.append(smi)
                    n_atoms.append(m.GetNumAtoms())
            except:
                pass

        gc.collect()

        pop_valid_smis, pop_valid_scores = [], []
        for m, s in zip(new_mating_pool, new_mating_scores):
            try:
                # pop_valid_smis.append(Chem.MolToSmiles(m))
                pop_valid_smis.append(Chem.MolToSmiles(m))
                pop_valid_scores.append(s)
            except:
                pass
        
        return smis, n_atoms, pop_valid_smis, pop_valid_scores
