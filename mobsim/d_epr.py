import pandas as pd
import numpy as np
import random

from base import get_waitTime, get_initUser, get_rhoP, get_gammaP, get_initLoc, load_data, base_model, post_process


np.random.seed(0)
random.seed(0)


def explore(all_loc, sp_p, pair_distance):
    """The exploration step of the density epr model."""
    curr_loc = int(all_loc[-1])
    curr_attr = sp_p[curr_loc]

    all_loc = set(all_loc)
    remain_idx = np.arange(sp_p.shape[0])
    remain_idx = np.delete(remain_idx, list(all_loc))

    r = (pair_distance[curr_loc, remain_idx] / 1000) ** (1.7)
    attr = np.power((sp_p[remain_idx] * curr_attr), 0.5).astype(float)
    attr = np.divide(attr, r, out=np.zeros_like(attr), where=r != 0)
    attr = attr / attr.sum()

    selected_loc = np.random.choice(attr.shape[0], p=attr)
    return remain_idx[selected_loc]


def preferential_return(hist_loc):
    # note: not able to return to the current location
    hist_loc = np.array(hist_loc)
    currloc_idx = np.where(hist_loc == hist_loc[-1])[0]
    locations = np.delete(hist_loc, currloc_idx)

    # new_hist_loc = hist_loc[hist_loc!=curr_loc]
    next_loc = locations[np.random.randint(low=0, high=len(locations))]

    return next_loc


def depr_model(sp_p, home_locs_df, pair_distance, sequence_length=10):
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

            if (np.random.rand() < if_explore) or (len(loc_ls) == 1):
                # explore
                next_loc = explore(curr_loc=loc_ls[-1], sp_p=sp_p, pair_distance=pair_distance)
            else:
                # return
                next_loc = preferential_return(hist_loc=loc_ls)

        # print(next_loc)
        loc_ls.append(next_loc)

    return loc_ls, dur_ls, user


if __name__ == "__main__":
    data_dir = "./WP1/npp/data"

    sp_p, all_locs, home_locs_df, pair_distance, _ = load_data(data_dir)

    syn_data = base_model(
        depr_model,
        population=100,
        sequence_length=3000,
        n_jobs=1,
        sp_p=sp_p,
        home_locs_df=home_locs_df,
        pair_distance=pair_distance,
    )

    save_dir = "./WP1/npp/data/out/generated_locs_depr.csv"
    post_process(syn_data, save_dir, all_locs)
