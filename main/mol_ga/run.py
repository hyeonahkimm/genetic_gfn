from __future__ import annotations

import joblib

from main.mol_ga.mol_ga import default_ga
from main.mol_ga.mol_ga.mol_libraries import random_zinc
from main.optimizer import BaseOptimizer



class MolGAOptimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "mol_ga"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)

        kwargs = dict(
            starting_population_smiles=random_zinc(1000),
            scoring_function=self.oracle,
            max_generations=config['max_generations'],
            offspring_size=config['offspring_size'],
        )

        if config['parallel']:
            with joblib.Parallel(n_jobs=-1) as parallel:
                output = default_ga(**kwargs, parallel=parallel)
        else:
            output = default_ga(**kwargs, parallel=None)