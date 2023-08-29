import argparse
import random

import numpy as np

from ipt import preferential_return, build_emperical_markov_matrix
from base import (
    get_waitTime,
    get_initUser,
    get_rhoP,
    get_gammaP,
    get_initLoc,
    load_data,
    base_model,
    post_process,
)
from d_epr import explore

np.random.seed(0)
random.seed(0)


class DTEpr:
    def __init__(self):
        pass

    def simulate(self):
        pass

    def simulate_agent(self):
        pass

    def simulate_agent_step(self):
        pass


def IPT_dEPR_model(sp_p, home_locs_df, user, pair_distance, emp_mat, sequence_length=10):
    trans_mat = np.copy(emp_mat)

    loc_ls = []
    dur_ls = []

    # get the two exploration parameter
    rho = get_rhoP()
    gamma = get_gammaP()

    # the generation process
    for _ in range(sequence_length):
        # get wait time from distribution - rather independent from other process
        curr_dur = get_waitTime()
        dur_ls.append(curr_dur)

        # init
        if len(loc_ls) == 0:
            next_loc = get_initLoc(home_locs_df, user)
        else:  # or generate
            # the prob. of exploring
            if_explore = rho * len(np.unique(loc_ls)) ** (-gamma)
            # print(if_explore)

            if (np.random.rand() < if_explore) or (len(loc_ls) == 1):
                # explore
                next_loc = explore(all_loc=loc_ls, sp_p=sp_p, pair_distance=pair_distance)
            else:
                trans_mat, next_loc = preferential_return(hist_loc=loc_ls, emp_mat=trans_mat)
                # next_loc = loc_ls[np.random.randint(low=0, high=len(loc_ls))]

        # print(next_loc)
        loc_ls.append(next_loc)

    del trans_mat
    return loc_ls, dur_ls, user


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="Path to the data file.",
        default="WP1/npp/data/input",
    )
    parser.add_argument(
        "file_name",
        type=str,
        nargs="?",
        help="Saving directory.",
        default="generated_ipt_2.csv",
    )
    parser.add_argument(
        "population",
        type=int,
        nargs="?",
        help="Population number to generate",
        default="800",
    )
    parser.add_argument(
        "sequence_length",
        type=int,
        nargs="?",
        help="Length of generated location sequence for each user",
        default="2000",
    )
    parser.add_argument(
        "if_prior",
        type=bool,
        nargs="?",
        help="If including GC transition as prior (IPT)",
        default="True",
    )
    parser.add_argument(
        "if_attract",
        type=bool,
        nargs="?",
        help="If including GC population attractiveness as prior (d-EPR)",
        default="True",
    )
    parser.add_argument(
        "n_jobs",
        type=int,
        nargs="?",
        default="1",
    )

    args = parser.parse_args()

    sp_p, all_locs, home_locs_df, pair_distance, all_sp = load_data(args.data_dir)
    if args.if_attract:
        # sp_p = np.ones_like(sp_p)
        index = np.argsort(sp_p)[::-1]

    emp_mat = build_emperical_markov_matrix(all_sp, prior=args.if_prior)

    user_arr = np.tile(home_locs_df["user_id"].values, args.population // 80 + 1)[: args.population]

    syn_data = base_model(
        IPT_dEPR_model,
        population=args.population,
        sequence_length=args.sequence_length,
        n_jobs=args.n_jobs,
        user_arr=user_arr,
        sp_p=sp_p,
        home_locs_df=home_locs_df,
        pair_distance=pair_distance,
        emp_mat=emp_mat,
    )

    post_process(syn_data, args.data_dir, args.file_name, all_locs)
