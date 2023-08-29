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


def preferential_return(hist_loc, emp_mat):
    # note: not able to return to the current location
    hist_loc = np.array(hist_loc)
    curr_loc = hist_loc[-1]

    # delete the current location from the sequence
    currloc_idx = np.where(hist_loc == hist_loc[-1])[0]
    hist_loc = np.delete(hist_loc, currloc_idx)

    # get the transition p frpm emperical matrix
    curr_trans_p = emp_mat[curr_loc, :]
    curr_trans_p = curr_trans_p[np.array(hist_loc)]

    # equal p if no prior knowledge
    if curr_trans_p.sum() == 0:
        curr_trans_p = np.ones([len(curr_trans_p)]) / len(curr_trans_p)
    else:
        curr_trans_p = curr_trans_p / curr_trans_p.sum()

    # choose next location according to emperical p
    next_loc = np.random.choice(hist_loc, p=curr_trans_p)
    # update
    emp_mat[curr_loc, next_loc] += 1

    return emp_mat, next_loc


def IPT_model(home_locs_df, pair_distance, emp_mat, sequence_length=10):
    trans_mat = np.copy(emp_mat)
    # get the user id
    user = get_initUser(home_locs_df)

    loc_ls = []
    dur_ls = []

    # get the two exploration parameter
    rho = get_rhoP()
    gamma = get_gammaP()
    # print(rho, gamma)
    emp_mat

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

            if np.random.rand() < if_explore:
                # explore
                next_loc = explore(curr_loc=loc_ls[-1], pair_distance=pair_distance)
            else:
                trans_mat, next_loc = preferential_return(hist_loc=loc_ls, emp_mat=trans_mat)
                # next_loc = loc_ls[np.random.randint(low=0, high=len(loc_ls))]

        # print(next_loc)
        loc_ls.append(next_loc)

    del trans_mat
    return loc_ls, dur_ls, user


def get_user_transition_ls(df):
    ls = df["location_id"].values

    res = []
    for i in range(len(ls) - 1):
        res.append([ls[i], ls[i + 1]])

    return res


def build_emperical_markov_matrix(sp_seq, prior=True):
    sp_seq["location_id"] = sp_seq["location_id"].astype(int)
    sp_seq = sp_seq[["user_id", "location_id"]]

    loc_size = sp_seq["location_id"].max() + 1
    trans_matrix = np.zeros([loc_size, loc_size])

    tran_ls = sp_seq.groupby("user_id").apply(get_user_transition_ls)
    tran_ls = [x for xs in tran_ls.values for x in xs]
    for tran in tran_ls:
        trans_matrix[tran[0], tran[1]] += 1

    # for i in range(len(trans_matrix)):
    #     print(trans_matrix[i])
    #     index = np.argsort(trans_matrix[i])[::-1]
    #     print(trans_matrix[i].sum())
    #     top_idx = index[:10]
    #     values = trans_matrix[i, top_idx]
    #     print(trans_matrix[i].sum())
    #     print(values)
    #     np.random.shuffle(values)
    #     print(values)
    #     trans_matrix[i, top_idx] = values

    #     print(trans_matrix[i])

    #     print(trans_matrix[i].sum())

    # if prior:
    # transform transition list to matrix

    return trans_matrix.astype(np.int16)


if __name__ == "__main__":
    data_dir = "./WP1/npp/data"

    _, all_locs, home_locs_df, pair_distance, all_sp = load_data(data_dir)

    emp_mat = build_emperical_markov_matrix(all_sp, prior=False)

    syn_data = base_model(
        IPT_model,
        population=100,
        sequence_length=3000,
        n_jobs=-1,
        home_locs_df=home_locs_df,
        pair_distance=pair_distance,
        emp_mat=emp_mat,
    )

    save_dir = "./WP1/npp/data/out/generated_locs_ipt.csv"
    post_process(syn_data, save_dir, all_locs)
