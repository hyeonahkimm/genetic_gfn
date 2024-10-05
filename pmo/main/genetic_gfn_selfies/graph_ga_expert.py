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

import selfies as sf
import main.genetic_gfn.genetic_operator.crossover as co
import main.genetic_gfn.genetic_operator.mutate as mu

import gc


MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, population_size: int, rank_coefficient=0.01, blended=False, frac_graph_ga_mutate=0.1, low_score_ratio=0.):
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
    if rank_coefficient >= 1:
        quantiles = 1 - np.logspace(-3, 0, 25)
        n_samples_per_quanitile = int(np.ceil(population_size / len(quantiles)))

        mating_pool, mating_pool_score = [], []
        for q in quantiles:
            score_threshold = np.quantile(population_scores, q)
            eligible_population = [(smiles, score) for score, smiles in zip(population_scores, population_mol) if score >= score_threshold]
            # samples.extend(np.random.choices(population=eligible_population, k=n_samples_per_quanitile))
            indices = np.random.choice(np.arange(len(eligible_population)), size=n_samples_per_quanitile)
            mating_pool.extend([eligible_population[i][0]for i in indices if eligible_population[i][0] is not None])
            mating_pool_score.extend([eligible_population[i][1] for i in indices if eligible_population[i][0] is not None])
    elif rank_coefficient > 0:
        scores_np = np.array(population_scores)
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)
        
        low_indices = []
        if low_score_ratio > 0:
            low_pop_size = int(population_size * low_score_ratio)
            score_ub = np.quantile(population_scores, 0.25)
            prob = (population_scores <= score_ub) / (population_scores <= score_ub).sum()
            low_indices = np.random.choice(np.arange(len(population_scores)), p=prob, size=low_pop_size).tolist()

        indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=population_size - len(low_indices), replacement=True
            ))
        mating_pool = [population_mol[i] for i in indices+low_indices if population_mol[i] is not None]
        mating_pool_score = [population_scores[i] for i in indices+low_indices if population_mol[i] is not None]
        # print(mating_pool)
    else:
        population_scores = [s + MINIMUM for s in population_scores]
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
        # mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
        indices = np.random.choice(np.arange(len(population_mol)), p=population_probs, size=population_size, replace=True)
        mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
        mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]

    if blended:
        mutate_mating_pool, crossover_mating_pool = [], []
        mutate_mating_score, crossover_mating_score = [], []
        for i in range(len(mating_pool)):
            if population_mol[i] is not None:
                if np.random.rand(1) < frac_graph_ga_mutate and len(mutate_mating_pool) < int(population_size * frac_graph_ga_mutate) + 1:
                    mutate_mating_pool.append(population_mol[i])
                    mutate_mating_score.append(population_scores[i])
                else:
                    crossover_mating_pool.append(population_mol[i])
                    crossover_mating_score.append(population_scores[i])
        return crossover_mating_pool, crossover_mating_score, mutate_mating_pool, mutate_mating_score

    return mating_pool, mating_pool_score, None, None


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
        self.chromosome = 'graph'
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def get_final_population(self, mating_pool, rank_coefficient=0.):
        new_mating_pool, new_mating_scores, _, _ = make_mating_pool(mating_pool[0], mating_pool[1], self.population_size, rank_coefficient)
        return (new_mating_pool, new_mating_scores)

    def query(self, query_size, mating_pool, pool, rank_coefficient=0.01, blended=False, mutation_rate=None, low_score_ratio=0.):
        # print(mating_pool)
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        cross_mating_pool, cross_mating_scores, mut_mating_pool, mut_mating_score = make_mating_pool(population_mol, population_scores, self.population_size, rank_coefficient, blended, low_score_ratio=low_score_ratio)

        if blended:
            mut_size = int(query_size * 0.1)
            cross_size = query_size - mut_size
            mut_offspring_mol = mu.mutate(mut_mating_pool, mutation_rate=0.01)
            cross_offspring_mol = pool(delayed(reproduce)(cross_mating_pool, mutation_rate) for _ in range(cross_size))
            offspring_mol = cross_offspring_mol + mut_offspring_mol
            new_mating_pool = cross_mating_pool + mut_mating_pool
            new_mating_scores = cross_mating_scores + mut_mating_score
        else:
            offspring_mol = pool(delayed(reproduce)(cross_mating_pool, mutation_rate) for _ in range(query_size))
            new_mating_pool = cross_mating_pool
            new_mating_scores = cross_mating_scores

        smiles, n_atoms = [], []
        selfies = []
        for m in offspring_mol:
            try:
                # smis.append(Chem.MolToSmiles(m))
                smi = Chem.MolToSmiles(m)
                encoded = sf.encoder(smi)
                if encoded not in selfies:  # unique
                    selfies.append(encoded)
                    smiles.append(smi)
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
        
        return selfies, smiles, pop_valid_smis, pop_valid_scores
