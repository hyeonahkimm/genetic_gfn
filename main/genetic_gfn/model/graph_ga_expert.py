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

import main.graph_ga.crossover as co, main.graph_ga.mutate as mu

import gc


MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int, rank_based=False, return_pop=False, replace=True):
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
    if rank_based:
        scores_np = np.array(population_scores)
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (1e-3 * len(scores_np) + ranks)
        indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=offspring_size, replacement=replace
            ))
        mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]
        # print(mating_pool)
    else:
        population_scores = [s + MINIMUM for s in population_scores]
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
        # mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
        indices = np.random.choice(np.arange(len(population_mol)), p=population_probs, size=offspring_size, replace=replace)
        mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]

    if return_pop:
        return mating_pool, mating_pool_score

    return mating_pool, None


def make_blended_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int, rank_based=False, frac_graph_ga_mutate=0.1, replace=True):
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
    if rank_based:
        scores_np = np.array(population_scores)
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (1e-3 * len(scores_np) + ranks)
        indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=offspring_size, replacement=replace
            ))
        # mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        
        # print(mating_pool)
    else:
        population_scores = [s + MINIMUM for s in population_scores]
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
        # mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
        indices = np.random.choice(np.arange(len(population_mol)), p=population_probs, size=offspring_size, replace=replace)

    mutate_mating_pool, crossover_mating_pool = [], []
    mutate_mating_score, crossover_mating_score = [], []
    for i in indices:
        if population_mol[i] is not None:
            if np.random.rand(1) < frac_graph_ga_mutate and len(mutate_mating_pool) < int(offspring_size * frac_graph_ga_mutate) + 1:
                mutate_mating_pool.append(population_mol[i])
                mutate_mating_score.append(population_scores[i])
            else:
                crossover_mating_pool.append(population_mol[i])
                crossover_mating_score.append(population_scores[i])

    # print(crossover_mating_pool)
    # crossover_mating_pool = random.shuffle(crossover_mating_pool)

    return mutate_mating_pool, crossover_mating_pool, mutate_mating_score, crossover_mating_score


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
    def __init__(self, mutation_rate: float=0.067, population_size=200, offspring_size=50, rank_based=False):
        self.chromosome = 'graph'
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.rank_based = rank_based

    def get_final_population(self, mating_pool, rank_based=False):
        new_mating_pool, new_mating_scores = make_mating_pool(mating_pool[0], mating_pool[1], self.population_size, rank_based, return_pop=True, replace=False)
        return (new_mating_pool, new_mating_scores)

    def query(self, query_size, mating_pool, pool, rank_based=True, return_pop=False):
        # print(mating_pool)
        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        new_mating_pool, new_mating_scores = make_mating_pool(population_mol, population_scores, self.population_size, rank_based, return_pop)

        offspring_mol = pool(delayed(reproduce)(new_mating_pool, self.mutation_rate) for _ in range(query_size))

        smis = []
        for m in offspring_mol:
            try:
                smis.append(Chem.MolToSmiles(m, canonical=True))
            except:
                pass

        gc.collect()

        if return_pop:
            pop_valid_smis, pop_valid_scores = [], []
            for m, s in zip(new_mating_pool, new_mating_scores):
                try:
                    pop_valid_smis.append(Chem.MolToSmiles(m, canonical=True))
                    pop_valid_scores.append(s)
                except:
                    pass
            return smis, pop_valid_smis, pop_valid_scores

        return smis, None, None

    def blended_query(self, query_size, mating_pool, pool, frac_graph_ga_mutate=0.1, rank_based=True, return_pop=False):
        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        mut_mating_pool, cross_mating_pool, mut_mating_score, cross_mating_score = make_blended_mating_pool(population_mol, population_scores, self.population_size, rank_based, frac_graph_ga_mutate, replace=True)

        mut_offspring_mol = mu.mutate(mut_mating_pool, mutation_rate=0.)
        cross_offspring_mol = pool(delayed(reproduce)(cross_mating_pool, self.mutation_rate) for _ in range(query_size))

        smis = []
        for m in mut_offspring_mol + cross_offspring_mol:
            try:
                smis.append(Chem.MolToSmiles(m, canonical=True))
            except:
                pass
        
        gc.collect()

        if return_pop:
            pop_valid_smis, pop_valid_scores = [], []
            for m, s in zip(mut_mating_pool+cross_mating_pool, mut_mating_score+cross_mating_score):
                try:
                    pop_valid_smis.append(Chem.MolToSmiles(m, canonical=True))
                    pop_valid_scores.append(s)
                except:
                    pass
            return smis, pop_valid_smis, pop_valid_scores
            
        return smis, None, None

