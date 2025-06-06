from __future__ import print_function

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 

def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='graph_ga')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"]) ### 
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument('--run_name', type=str, default="default")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb

    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"

    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones
    if args.method == 'graph_ga':
        from main.graph_ga.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'smiles_ga':
        from main.smiles_ga.run import SMILES_GA_Optimizer as Optimizer
    elif args.method == "selfies_ga":
        from main.selfies_ga.run import SELFIES_GA_Optimizer as Optimizer
    elif args.method == "synnet":
        from main.synnet.run import SynNet_Optimizer as Optimizer
    elif args.method == "smiles_lstm_hc":
        from main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'selfies_lstm_hc':
        from main.selfies_lstm_hc.run import SELFIES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'gegl':
        from main.gegl.run import GEGL_Optimizer as Optimizer
    elif args.method == 'gpbo':
        from main.gpbo.run import GPBO_Optimizer as Optimizer
    elif args.method == 'stoned': 
        from main.stoned.run import Stoned_Optimizer as Optimizer
    elif args.method == 'gflownet':
        from main.gflownet.run import GFlowNet_Optimizer as Optimizer
    elif args.method == 'gflownet_al':
        from main.gflownet_al.run import GFlowNet_AL_Optimizer as Optimizer
    elif args.method == 'reinvent':
        from main.reinvent.run import REINVENT_Optimizer as Optimizer
    elif args.method == 'reinvent_selfies':
        from main.reinvent_selfies.run import REINVENT_SELFIES_Optimizer as Optimizer
    elif args.method == "mol_ga":
        from main.mol_ga.run import MolGAOptimizer as Optimizer
    elif args.method == "genetic_gfn":
        from main.genetic_gfn.run import Genetic_GFN_Optimizer as Optimizer
    elif args.method == "genetic_gfn_selfies":
        from main.genetic_gfn_selfies.run import Genetic_GFN_SELFIES_Optimizer as Optimizer
    elif args.method == "genetic_gfn_al":
        from main.genetic_gfn_al.run import Genetic_GFN_AL_Optimizer as Optimizer
    elif args.method == "reinvent_ls_gfn":
        from main.genetic_gfn.run_ls_gfn import REINVENT_LS_GFN_Optimizer as Optimizer
        path_main = os.path.dirname(os.path.realpath(__file__))
        path_main = os.path.join(path_main, "main", "genetic_gfn")
    else:
        raise ValueError("Unrecognized method name.")


    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main

    if args.task != "tune":
    
        for oracle_name in args.oracles:

            print(f'Optimizing oracle function: {oracle_name}')

            try:
                config_default = yaml.safe_load(open(args.config_default))
            except:
                config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

            oracle = Oracle(name = oracle_name)
            optimizer = Optimizer(args=args)

            if args.task == "simple":
                # optimizer.optimize(oracle=oracle, config=config_default, seed=args.seed) 
                for seed in args.seed:
                    print('seed', seed)
                    optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
            elif args.task == "production":
                optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
            else:
                raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')

    elif args.task == "tune":

        print(f'Tuning hyper-parameters on tasks: {args.oracles}')

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

        try:
            config_tune = yaml.safe_load(open(args.config_tune))
        except:
            config_tune = yaml.safe_load(open(os.path.join(path_main, args.config_tune)))

        oracles = [Oracle(name = oracle_name) for oracle_name in args.oracles]
        optimizer = Optimizer(args=args)
        
        optimizer.hparam_tune(oracles=oracles, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)

    else:
        raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')
    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    main()

