from __future__ import print_function

import torch
import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from polyleven import levenshtein
from tdc import Evaluator
rdBase.DisableLog('rdApp.error')

import main.genetic_gfn.genetic_operator.crossover as co
import main.genetic_gfn.genetic_operator.mutate as mu

from tdc.chem_utils import MolConvert
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')
smiles2selfies = MolConvert(src = 'SMILES', dst = 'SELFIES')

from main.stoned.run import get_selfie_chars, mutate_selfie

import gc


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
    return new_child, parent_a, parent_b


class GeneticOperatorHandler:
    def __init__(self, mutation_rate: float=0.067, population_size=200):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.mol_distance_func = Evaluator(name = 'Diversity')
        self.mol_novelty_func = Evaluator(name = 'Novelty')
    
    def smiles_distance_func(self, smi_a, smi_b):
        return levenshtein(smi_a, smi_b) / max(len(smi_a), len(smi_b))

    def query(self, query_size, mating_pool, pool, rank_coefficient=0.01, mutation_rate=None, return_dist=False):
        # print(mating_pool)
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        cross_mating_pool, cross_mating_scores = make_mating_pool(population_mol, population_scores, self.population_size, rank_coefficient)

        offspring_mol, parents_a, parents_b = zip(*[reproduce(cross_mating_pool, mutation_rate) for _ in range(query_size)])
        new_mating_pool = cross_mating_pool
        new_mating_scores = cross_mating_scores

        smis, n_atoms = [], []
        # rst = {}
        smiles_distances, smiles_novelty = [], []
        mol_distances, mol_novelty = [], []
        parents_smis, chileren_smis = [], []
        for m, a, b in zip(offspring_mol, parents_a, parents_b):
            try:
                # smis.append(Chem.MolToSmiles(m))
                smi = Chem.MolToSmiles(m)
                if return_dist:
                    smi_a = Chem.MolToSmiles(a)
                    smi_b = Chem.MolToSmiles(b)
                    chileren_smis.append(smi)
                    parents_smis.append((smi_a, smi_b))
                    smiles_distances.append((levenshtein(smi_a, smi) + levenshtein(smi_b, smi)) / 2)
                    smiles_novelty.append(min(levenshtein(smi_a, smi), levenshtein(smi_b, smi)))
                    mol_distances.append((self.mol_distance_func([smi_a, smi]) + self.mol_distance_func([smi_b, smi]))/2)
                    mol_novelty.append(self.mol_novelty_func([smi], [smi_a, smi_b]))
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

        if return_dist:
            results = {'children': chileren_smis, 'parents': parents_smis, 'smiles_dist': smiles_distances, 'mol_dist': mol_distances, 'smiles_novelty': smiles_novelty, 'mol_novelty': mol_novelty}
            return smis, n_atoms, pop_valid_smis, pop_valid_scores, results
        
        return smis, n_atoms, pop_valid_smis, pop_valid_scores



class STONED:
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
