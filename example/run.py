import argparse

from mobsim import Environment
from mobsim import DTEpr, DEpr, IPT


if __name__ == "__main__":
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
    simulator = IPT(env)

    traj = simulator.simulate(seq_len=args.seq_len, pop_num=args.pop_num)

    print(traj)
