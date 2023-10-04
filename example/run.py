import argparse

import numpy as np
import random

from mobsim import Environment
from mobsim import DTEpr, DEpr, IPT, EPR


def setup_seed(seed):
    """
    fix random seed for reproducing
    """
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    setup_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pop_num",
        type=int,
        nargs="?",
        help="Population number to generate",
        default="100",
    )
    parser.add_argument(
        "seq_len",
        type=int,
        nargs="?",
        help="Length of generated location sequence for each user",
        default="20",
    )
    parser.add_argument(
        "model",
        default="dtepr",
        nargs="?",
        choices=["epr", "ipt", "depr", "dtepr"],
        help="Individual mobility model for generation (default: %(default)s)",
    )
    args = parser.parse_args()

    env = Environment("./example/config.yml")
    if args.model == "epr":
        simulator = EPR(env)
    elif args.model == "ipt":
        simulator = IPT(env)
    elif args.model == "depr":
        simulator = DEpr(env)
    elif args.model == "dtepr":
        simulator = DTEpr(env)
    else:
        raise AttributeError(
            f"Model unknown. Please check the input arguement. We only support 'epr', 'ipt', 'depr', 'dtepr'. You passed {args.model}"
        )

    traj = simulator.simulate(seq_len=args.seq_len, pop_num=args.pop_num)

    print(traj)

    traj.to_csv(f"data/output/{args.model}.csv")
