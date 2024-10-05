
# import rootutils
# root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import argparse
from genetic_gfn.train_agent import train_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_prior_from", type=str, default='data/Prior.ckpt')
    parser.add_argument("--sigma", type=float, default=500)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("--run", default=0, help="run", type=int)
    parser.add_argument('--save', action='store_true',
                        default=False, help='Save model.')
    parser.add_argument('--debug',action='store_true',
                        default=False, help='debug mode, no multi thread')
    parser.add_argument("--enable_tensorboard",
                        action='store_true', default=False)
    parser.add_argument("--n_steps", default=10, type=int)
    parser.add_argument("--experience_replay", default=24, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--log_weight_score", action='store_true', default=False)
    parser.add_argument("--max_oracle_calls", default=1000, type=int)
    parser.add_argument("--freq_log", default=100, type=int)

    # objectives
    parser.add_argument("--objectives", type=str,
                        default='gsk3b,jnk3,qed,sa')
    parser.add_argument("--scalar", default='WeightedSum', type=str)
    parser.add_argument("--alpha", default=1., type=float,
                        help='dirichlet distribution')
    parser.add_argument("--alpha_vector", default='1,1,1,1', type=str)
    args = parser.parse_args()
    args.objectives = args.objectives.split(',')
    args.alpha_vector = args.alpha_vector.split(',')
    args.alpha_vector = [float(x) for x in args.alpha_vector]
    
    train_agent(args)
