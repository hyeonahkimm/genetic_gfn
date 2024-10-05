from __future__ import print_function

import torch
from typing import List

import numpy as np
from rdkit import rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

from tdc.chem_utils import MolConvert
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')
smiles2selfies = MolConvert(src = 'SMILES', dst = 'SELFIES')

from main.stoned.run import get_selfie_chars, mutate_selfie


MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, population_size: int, rank_coefficient=0.01):
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




class GeneticOperatorHandler:
    def __init__(self, mutation_rate: float=0.067, population_size=200):
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def query(self, query_size, mating_pool, pool, rank_coefficient=0.01, mutation_rate=None, return_dist=False):
        # print(mating_pool)
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        population_mol = [smiles2selfies(s) for s in mating_pool[0]]  # smiles -> selfie  # [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        # rank-based
        # cross_mating_pool, cross_mating_scores = make_mating_pool(population_mol, population_scores, self.population_size, rank_coefficient)
        
        len_random_struct = max([len(get_selfie_chars(s)) for s in population_mol])  # smiles -> selfie
        
        #    Step 1: Keep the best molecule:  Keep the best member & mutate the rest
        best_idx = np.argmax(population_scores)
        best_selfie = population_mol[best_idx]
        
        #    Step 2: Get mutated selfies 
        new_population = []
        for i in range(query_size-1): 
            selfie_mutated, _ = mutate_selfie(best_selfie, len_random_struct, write_fail_cases=True) 
            new_population.append(selfie_mutated)
        new_population.append(best_selfie)
        
        # selfies -> smiles
        new_smiles = [selfies2smiles(s) for s in new_population]
        
        return new_smiles, None, None, None, None
