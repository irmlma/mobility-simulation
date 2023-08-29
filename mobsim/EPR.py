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


def epr_model_individual(home_locs_df, pair_distance, sequence_length=10):
    # get the user id
    user = get_initUser(home_locs_df)

    loc_ls = []
    dur_ls = []

    # get the two exploration parameter
    rho = get_rhoP()
    gamma = get_gammaP()
    # print(rho, gamma)

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
                next_loc = loc_ls[np.random.randint(low=0, high=len(loc_ls))]

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

    save_dir = "./WP1/npp/data/out/generated_locs_epr.csv"
    post_process(syn_data, save_dir, all_locs)
