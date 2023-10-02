import pandas as pd
import geopandas as gpd

import numpy as np


from mobsim.env import Environment
from scipy.spatial.distance import pdist, squareform


class EPR:
    """Explore and preferential return model"""

    def __init__(self, env: Environment):
        self.env = env

        # precalculate distance between location pairs
        coord_list = [[x, y] for x, y in zip(self.env.loc_gdf["center"].x, self.env.loc_gdf["center"].y)]
        Y = pdist(coord_list, "euclidean")
        self.pair_distance = squareform(Y)

        self.traj = {}

    def simulate(self, seq_len=20, pop_num=100):
        # get the user_id for generation
        user_arr = np.tile(self.env.top_user_loc_df["user_id"].values, pop_num // 80 + 1)[:pop_num]

        for i in range(pop_num):
            res = self.simulate_agent(user_arr[i], seq_len)

            self.traj[i] = {}
            self.traj[i]["user"] = user_arr[i]
            self.traj[i]["loc_seq"], self.traj[i]["dur_seq"] = res

        return post_process(self.traj, self.env.loc_gdf)

    def simulate_agent(self, user, seq_len):
        loc_ls = []
        dur_ls = []

        # get the two exploration parameter
        rho = self.env.get_rho()
        gamma = self.env.get_gamma()

        # the generation process
        for _ in range(seq_len):
            # get wait time from distribution
            dur_ls.append(self.env.get_wait_time())

            next_loc = self.simulate_agent_step(user, loc_ls, rho, gamma)

            loc_ls.append(next_loc)

        return loc_ls, dur_ls

    def simulate_agent_step(self, user, loc_ls, rho, gamma):
        if len(loc_ls) == 0:  # init
            next_loc = self.get_init_loc(user)
        else:  # or generate
            # the prob. of exploring
            if_explore = rho * len(np.unique(loc_ls)) ** (-gamma)

            if (np.random.rand() < if_explore) or (len(loc_ls) == 1):
                # explore
                next_loc = self.explore(visited_loc=loc_ls)
            else:
                next_loc = self.pref_return(visited_loc=loc_ls)

        return next_loc

    def get_init_loc(self, user):
        """The initialization step of the epr model.

        Currently we choose one of the top5 visted location as the start.

        """
        candidate = self.env.top_user_loc_df.loc[self.env.top_user_loc_df["user_id"] == user, "location_id"].values
        return int(np.random.choice(candidate))

    def explore(self, visited_loc):
        """The exploration step of the epr model.

        1. get a jump distance from predefined jump length distribution.
        2. calculate the distance between current location and all the other locations.
        3. choose the location that is closest to the jump length.

        Parameters
        ----------
        curr_loc: the current location that the user is standing
        all_loc: df containing the info of all locations

        Returns
        -------
        the id of the selected location
        """
        curr_loc = int(visited_loc[-1])
        # the distance to be jumped
        jump_distance = self.env.get_jump()

        # select the closest location after the jump
        return np.argsort(np.abs(self.pair_distance[curr_loc, :] - jump_distance))[0]

    def pref_return(self, visited_loc):
        # not able to return to the current location
        visited_loc = np.array(visited_loc)
        curr_loc = visited_loc[-1]

        # delete the current location from the sequence
        currloc_idx = np.where(visited_loc == curr_loc)[0]
        # ensure the deleted sequence contain value
        if len(currloc_idx) != len(visited_loc):
            visited_loc = np.delete(visited_loc, currloc_idx)

        # choose next location according to emperical visits
        next_loc = np.random.choice(visited_loc)

        return next_loc


class DTEpr(EPR):
    """Density Transition EPR"""

    def __init__(self, env: Environment, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.trans_matrix = self.build_emperical_markov_matrix()

    def build_emperical_markov_matrix(self):
        # initialize trans matrix with 0's
        loc_size = self.env.loc_seq_df["location_id"].max() + 1
        trans_matrix = np.zeros([loc_size, loc_size])

        def _get_user_transition_ls(df):
            ls = df["location_id"].values

            res = []
            for i in range(len(ls) - 1):
                res.append([ls[i], ls[i + 1]])

            return res

        # transform transition list to matrix
        tran_ls = self.env.loc_seq_df.groupby("user_id").apply(_get_user_transition_ls)
        tran_ls = [x for xs in tran_ls.values for x in xs]
        for tran in tran_ls:
            trans_matrix[tran[0], tran[1]] += 1

        return trans_matrix.astype(np.int16)

    def simulate_agent_step(self, user, loc_ls, rho, gamma):
        if len(loc_ls) == 0:  # init
            next_loc = self.get_init_loc(user)
        else:  # or generate
            # the prob. of exploring
            if_explore = rho * len(np.unique(loc_ls)) ** (-gamma)
            # print(if_explore)

            if (np.random.rand() < if_explore) or (len(loc_ls) == 1):
                # explore
                next_loc = self.explore(visited_loc=loc_ls)
            else:
                self.trans_matrix, next_loc = self.pref_return(visited_loc=loc_ls, emp_mat=self.trans_matrix)

        return next_loc

    def explore(self, visited_loc):
        """The exploration step of the density epr model."""
        attr = self.env.loc_gdf.copy()["count"].values

        curr_loc = int(visited_loc[-1])
        curr_attr = attr[curr_loc]

        # delete the already visited locations
        all_loc = set(visited_loc)
        remain_idx = np.arange(attr.shape[0])
        remain_idx = np.delete(remain_idx, list(all_loc))

        # slight modification, original **2, and we changed to 1.7
        r = (self.pair_distance[curr_loc, remain_idx] / 1000) ** (1.7)
        # we also take the square root to reduce the density effect, otherwise too strong
        attr = np.power((attr[remain_idx] * curr_attr), 0.5).astype(float)

        # the density attraction + inverse distance
        attr = np.divide(attr, r, out=np.zeros_like(attr), where=r != 0)

        # norm
        attr = attr / attr.sum()
        selected_loc = np.random.choice(attr.shape[0], p=attr)
        return remain_idx[selected_loc]

    def pref_return(self, visited_loc, emp_mat):
        # not able to return to the current location
        visited_loc = np.array(visited_loc)
        curr_loc = visited_loc[-1]

        # delete the current location from the sequence
        currloc_idx = np.where(visited_loc == curr_loc)[0]
        # ensure the deleted sequence contain value
        if len(currloc_idx) != len(visited_loc):
            visited_loc = np.delete(visited_loc, currloc_idx)

        # get the transition p from emperical matrix
        curr_trans_p = emp_mat[curr_loc, :]
        curr_trans_p = curr_trans_p[np.array(visited_loc)]

        # equal p if no prior knowledge
        if curr_trans_p.sum() == 0:
            curr_trans_p = np.ones_like(curr_trans_p) / len(curr_trans_p)
        else:
            curr_trans_p = curr_trans_p / curr_trans_p.sum()

        # choose next location according to emperical p
        next_loc = np.random.choice(visited_loc, p=curr_trans_p)
        # update
        emp_mat[curr_loc, next_loc] += 1

        return emp_mat, next_loc


class DEpr(EPR):
    """Density EPR"""

    def __init__(self, env: Environment, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.env = env

    def explore(self, visited_loc):
        """The exploration step of the density epr model."""
        attr = self.env.loc_gdf.copy()["count"].values

        curr_loc = int(visited_loc[-1])
        curr_attr = attr[curr_loc]

        # delete the already visited locations
        all_loc = set(visited_loc)
        remain_idx = np.arange(attr.shape[0])
        remain_idx = np.delete(remain_idx, list(all_loc))

        # slight modification, original **2, and we changed to 1.7
        r = (self.pair_distance[curr_loc, remain_idx] / 1000) ** (1.7)
        # we also take the square root to reduce the density effect, otherwise too strong
        attr = np.power((attr[remain_idx] * curr_attr), 0.5).astype(float)

        # the density attraction + inverse distance
        attr = np.divide(attr, r, out=np.zeros_like(attr), where=r != 0)

        # norm
        attr = attr / attr.sum()
        selected_loc = np.random.choice(attr.shape[0], p=attr)
        return remain_idx[selected_loc]


class IPT(EPR):
    """Individual Preferential Transition model"""

    def __init__(self, env: Environment, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.trans_matrix = self.build_emperical_markov_matrix()

    def build_emperical_markov_matrix(self):
        # initialize trans matrix with 0's
        loc_size = self.env.loc_seq_df["location_id"].max() + 1
        trans_matrix = np.zeros([loc_size, loc_size])

        def _get_user_transition_ls(df):
            ls = df["location_id"].values

            res = []
            for i in range(len(ls) - 1):
                res.append([ls[i], ls[i + 1]])

            return res

        # transform transition list to matrix
        tran_ls = self.env.loc_seq_df.groupby("user_id").apply(_get_user_transition_ls)
        tran_ls = [x for xs in tran_ls.values for x in xs]
        for tran in tran_ls:
            trans_matrix[tran[0], tran[1]] += 1

        return trans_matrix.astype(np.int16)

    def simulate_agent_step(self, user, loc_ls, rho, gamma):
        if len(loc_ls) == 0:  # init
            next_loc = self.get_init_loc(user)
        else:  # or generate
            # the prob. of exploring
            if_explore = rho * len(np.unique(loc_ls)) ** (-gamma)

            if (np.random.rand() < if_explore) or (len(loc_ls) == 1):
                # explore
                next_loc = self.explore(visited_loc=loc_ls)
            else:
                self.trans_matrix, next_loc = self.pref_return(visited_loc=loc_ls, emp_mat=self.trans_matrix)

        return next_loc

    def pref_return(self, visited_loc, emp_mat):
        # not able to return to the current location
        visited_loc = np.array(visited_loc)
        curr_loc = visited_loc[-1]

        # delete the current location from the sequence
        currloc_idx = np.where(visited_loc == curr_loc)[0]

        # ensure the deleted sequence contain value
        if len(currloc_idx) != len(visited_loc):
            visited_loc = np.delete(visited_loc, currloc_idx)

        # get the transition p from emperical matrix
        curr_trans_p = emp_mat[curr_loc, :]
        curr_trans_p = curr_trans_p[np.array(visited_loc)]

        # equal p if no prior knowledge
        if curr_trans_p.sum() == 0:
            curr_trans_p = np.ones_like(curr_trans_p) / len(curr_trans_p)
        else:
            curr_trans_p = curr_trans_p / curr_trans_p.sum()

        # choose next location according to emperical p
        next_loc = np.random.choice(visited_loc, p=curr_trans_p)
        # update
        emp_mat[curr_loc, next_loc] += 1

        return emp_mat, next_loc


def post_process(traj, loc_gdf):
    def _get_result_user_df(user_seq, loc_gdf):
        user_seq_df = pd.DataFrame(user_seq["loc_seq"], columns=["id"])
        user_seq_df["duration"] = user_seq["dur_seq"]
        user_seq_df["ori_user_id"] = user_seq["user"]

        return_gdf = user_seq_df.reset_index().merge(loc_gdf, on="id", sort=False)

        # preserve simulation order
        return return_gdf.sort_values("index").reset_index(drop=True).drop(columns="index")

    all_ls = []
    for i in range(len(traj)):
        user_df = _get_result_user_df(traj[i], loc_gdf)
        user_df["user_id"] = i
        all_ls.append(user_df)

    all_df = pd.concat(all_ls).reset_index()

    all_gdf = gpd.GeoDataFrame(all_df, geometry="center", crs="EPSG:2056")
    all_gdf = all_gdf.to_crs("EPSG:4326")

    return all_gdf
