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
    args = parser.parse_args()

    env = Environment("./example/config.yml")
    simulator = DTEpr(env)

    traj = simulator.simulate(seq_len=args.seq_len, pop_num=args.pop_num)

    print(traj)
