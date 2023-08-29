import pandas as pd
import numpy as np
import random

from base import (
    get_waitTime,
    get_initUser,
    get_rhoP,
    get_gammaP,
    get_initLoc,
    load_data,
    base_model,
    get_jump,
    post_process,
)


np.random.seed(0)
random.seed(0)


def explore(curr_loc, pair_distance):
    """The exploration step of the epr model.

    1. get a jump distance from predefined jump length distribution.
    2. calculate the distance between current location and all the other locations.
    3. choose the location that is closest to the jump length.

    potential problem:
    1. location set from emperical dataset: locations that do not exist in dataset will not appear in synthesized dataset.
    2. all locations are regarded equal important for each individual: in the process of choosing a location,
    we do not include emperical info of whether the user has visited the location

    Parameters
    ----------
    curr_loc: the current location that the user is standing
    all_loc: df containing the info of all locations

    Returns
    -------
    the id of the selected location
    """
    curr_loc = int(curr_loc)
    # the distance to be jumped
    jump_distance = get_jump()
    # print("Jump length:", jump_distance)

    # select the closest location after the jump
    selected_loc = np.argsort(np.abs(pair_distance[curr_loc, :] - jump_distance))[0]
    return selected_loc


def recency(loc_ls):
    top40_p = np.array(
        [
            0.1554,
            0.2678,
            0.128,
            0.0936,
            0.0637,
            0.0427,
            0.0287,
            0.0216,
            0.0178,
            0.0145,
            0.0117,
            0.0107,
            0.0098,
            0.0092,
            0.0081,
            0.0079,
            0.0068,
            0.0068,
            0.0064,
            0.0059,
            0.0055,
            0.0052,
            0.0053,
            0.0048,
            0.0049,
            0.0052,
            0.0046,
            0.0044,
            0.0047,
            0.004,
            0.0039,
            0.0041,
            0.0037,
            0.0035,
            0.0038,
            0.0035,
            0.0032,
            0.0034,
            0.0027,
            0.0027,
        ]
    )
    top40_p = top40_p / top40_p.sum()

    hist = 40 if len(loc_ls) > 40 else len(loc_ls)
    idx = np.random.choice(a=hist, p=top40_p[:hist] / top40_p[:hist].sum())
    return loc_ls[::-1][idx]


def epr_model_individual(home_locs_df, pair_distance, sequence_length=10):
    # get the user id
    user = get_initUser(home_locs_df)

    loc_ls = []
    dur_ls = []

    # get the two exploration parameter
    rho = get_rhoP()
    gamma = get_gammaP()
    # print(rho, gamma)

    # for recency
    alpha = 0.1

    # the generation process
    for i in range(sequence_length):
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

            if np.random.rand() < if_explore:
                # explore
                curr_loc = loc_ls[-1]
                next_loc = explore(curr_loc, pair_distance)
            else:
                # recency
                if np.random.rand() < alpha:
                    # return, randomly choose one existing location
                    next_loc = loc_ls[np.random.randint(low=0, high=len(loc_ls))]
                else:
                    # recency
                    next_loc = recency(loc_ls)

        # print(next_loc)
        loc_ls.append(next_loc)

    return loc_ls, dur_ls, user


if __name__ == "__main__":
    data_dir = "./WP1/npp/data"

    _, all_locs, home_locs_df, pair_distance, _ = load_data(data_dir)

    syn_data = base_model(
        epr_model_individual,
        population=100,
        sequence_length=3000,
        n_jobs=-1,
        home_locs_df=home_locs_df,
        pair_distance=pair_distance,
    )

    save_dir = "./WP1/npp/data/out/generated_locs_repr.csv"
    post_process(syn_data, save_dir, all_locs)
