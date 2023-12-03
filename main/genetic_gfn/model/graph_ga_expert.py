from __future__ import print_function

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

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
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
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


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
    def __init__(self, mutation_rate: float=0.067, population_size = 120, offspring_size=70):
        self.chromosome = 'graph'
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def query(self, query_size, mating_pool, pool):
        # print(mating_pool)
        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        new_mating_pool = make_mating_pool(population_mol, population_scores, self.population_size)
        offspring_mol = pool(delayed(reproduce)(new_mating_pool, self.mutation_rate) for _ in range(query_size))

        # add new_population
        # population_mol += offspring_mol

        smis = [Chem.MolToSmiles(m, canonical=True) for m in offspring_mol]
        # smis = random.choices(mating_pool, k=query_size * 2)
        # smi0s, smi1s = smis[:query_size], smis[query_size:]
        # smis = pool(
        #     delayed(reproduce)(smi0, smi1, self.mutation_rate) for smi0, smi1 in zip(smi0s, smi1s)
        # )
        # smis = list(filter(lambda smi: smi is not None, smis))
        gc.collect()

        return smis