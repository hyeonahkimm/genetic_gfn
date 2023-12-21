from __future__ import print_function

import torch
import random
from typing import List

import nltk
import copy
import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

# import main.reinvent_ga.genetic_operator.crossover as co
# import main.reinvent_ga.genetic_operator.mutate as mu
from main.reinvent_ga.smiles_ga_operator.cfg_util import encode, decode
from main.reinvent_ga.smiles_ga_operator.smiles_grammar import GCFG
from main.reinvent_ga.utils import Variable, seq_to_smiles

import gc


MINIMUM = 1e-10

def cfg_to_gene(prod_rules, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [np.random.randint(0, 256)
                           for _ in range(max_len - len(gene))]
    return gene


def gene_to_cfg(gene):
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:
            break
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                     GCFG.productions()[rule].rhs())
        stack.extend(list(rhs)[::-1])
    return prod_rules


def mutation(gene):
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant


def mutate(p_gene):
    c_gene = mutation(p_gene)
    # c_smiles = canonicalize(cfg_util.decode(gene_to_cfg(c_gene)))
    return c_gene



def make_mating_pool(population_mol, population_scores, population_size, rank_coefficient=0.01):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_smis: list of smiles
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


# def reproduce(mating_pool, mutation_rate):
#     """
#     Args:
#         mating_pool: list of RDKit Mol
#         mutation_rate: rate of mutation
#     Returns:
#     """
#     parent_a = random.choice(mating_pool)
#     parent_b = random.choice(mating_pool)
#     new_child = co.crossover(parent_a, parent_b)
#     if new_child is not None:
#         new_child = mu.mutate(new_child, mutation_rate)
#     return new_child


class GeneticOperatorHandler:
    def __init__(self, mutation_rate: float=0.067, population_size=200, voc=None):
        self.chromosome = 'graph'
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.voc = voc

    def get_final_population(self, mating_pool, rank_coefficient=0.):
        new_mating_pool, new_mating_scores, _, _ = make_mating_pool(mating_pool[0], mating_pool[1], self.population_size, rank_coefficient)
        return (new_mating_pool, new_mating_scores)

    def query(self, query_size, mating_pool, pool, rank_coefficient=0.01, blended=False, mutation_rate=None):
        # print(mating_pool)
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        # population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        # population_scores = mating_pool[1]

        pop_smis, pop_scores = make_mating_pool(mating_pool[0], mating_pool[1], self.population_size, rank_coefficient)

        # print(pop_smis)
        # tokenized = self.voc.tokenize(smile)
        # encoded.append(Variable(self.voc.encode(tokenized)))

        population_genes = []
        for i, smi in enumerate(pop_smis):
            try:
                tokenized = self.voc.tokenize(smi)
                population_genes.append(Variable(self.voc.encode(tokenized)))
                # valid_pop_scores.append(scores[i])
            except:
                pass

        # population_genes = [Variable(self.voc.encode(self.voc.tokenize(s))) for s in pop_smis]
        # print(len(population_genes))

        choice_indices = np.random.choice(len(population_genes), query_size, replace=True)
        genes_to_mutate = [population_genes[i] for i in choice_indices]
        children = pool(delayed(mutate)(g) for g in genes_to_mutate)
        # print(children)

        # smis = seq_to_smiles(children, self.voc)

        # print(children) seqs.cpu().numpy()
        smis = []
        for c_gene in children:
            try:
                seq = c_gene.cpu().numpy()
                smis.append(self.voc.decode(seq))
            except:
                pass
        # for m in offspring_mol:
        #     try:
        #         # smis.append(Chem.MolToSmiles(m))
        #         smis.append(Chem.MolToSmiles(m, canonical=True))
        #     except:
        #         pass

        gc.collect()

        # pop_valid_smis, pop_valid_scores = [], []
        # for m, s in zip(new_mating_pool, new_mating_scores):
        #     try:
        #         # pop_valid_smis.append(Chem.MolToSmiles(m))
        #         pop_valid_smis.append(Chem.MolToSmiles(m, canonical=True))
        #         pop_valid_scores.append(s)
        #     except:
        #         pass
        return smis, pop_smis, pop_scores
