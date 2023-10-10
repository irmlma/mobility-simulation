import pandas as pd
import geopandas as gpd

import numpy as np
from tqdm import tqdm

from mobsim.env import Environment
from scipy.spatial.distance import pdist, squareform

from trackintel.geogr.distances import calculate_distance_matrix


class EPR:
    """Explore and preferential return model"""

    def __init__(self, env: Environment):
        self.env = env

        # precalculate distance between location pairs
        self.pair_distance = calculate_distance_matrix(self.env.loc_gdf, dist_metric="haversine")

        self.traj = {}

    def simulate(self, seq_len=20, pop_num=100):
        # get the user_id for generation
        user_arr = np.random.choice(self.env.top_user_loc_df["user_id"].unique(), pop_num)

        for i in tqdm(range(pop_num), desc="Generating users"):
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
            if self.env.config["P"] != 0:
                # hard intervention to P
                if_explore = self.env.config["P"]
            else:
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


class DEpr(EPR):
    """Density EPR"""

    def __init__(self, env: Environment, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        # constructing the location visitation frequency for density epr
        loc_freq = self.env.loc_gdf.set_index("id").join(
            self.env.loc_seq_df.groupby("location_id").size().to_frame("freq")
        )
        # we assign never visited locations a small value, such that they can also be visited
        self.attr = loc_freq.fillna(0.1)["freq"].values

    def explore(self, visited_loc):
        """The exploration step of the density epr model."""

        curr_loc = int(visited_loc[-1])
        curr_attr = self.attr[curr_loc]

        # delete the already visited locations
        uniq_visited_loc = set(visited_loc)
        remain_idx = np.arange(self.attr.shape[0])
        remain_idx = np.delete(remain_idx, list(uniq_visited_loc))

        # slight modification, original **2, and we changed to 1.7
        r = (self.pair_distance[curr_loc, remain_idx] / 1000) ** (1.7)
        # we also take the square root to reduce the density effect, otherwise too strong
        attr = np.power((self.attr[remain_idx] * curr_attr), 0.5).astype(float)

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
        loc_size = self.env.loc_gdf["id"].max() + 1
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

        return trans_matrix.astype(np.int32)

    def pref_return(self, visited_loc):
        # not able to return to the current location
        visited_loc = np.array(visited_loc)
        curr_loc = visited_loc[-1]

        # delete the current location from the sequence
        currloc_idx = np.where(visited_loc == curr_loc)[0]
        # ensure the deleted sequence contain value
        if len(currloc_idx) != len(visited_loc):
            visited_loc = np.delete(visited_loc, currloc_idx)

        # get the transition p from emperical matrix
        curr_trans_p = self.trans_matrix[curr_loc, :]
        curr_trans_p = curr_trans_p[np.array(visited_loc)]

        # equal p if no prior knowledge
        if curr_trans_p.sum() == 0:
            curr_trans_p = np.ones_like(curr_trans_p) / len(curr_trans_p)
        else:
            curr_trans_p = curr_trans_p / curr_trans_p.sum()

        # choose next location according to emperical p
        next_loc = np.random.choice(visited_loc, p=curr_trans_p)
        # update
        self.trans_matrix[curr_loc, next_loc] += 1

        return next_loc


class DTEpr(EPR):
    """Density Transition EPR"""

    def __init__(self, env: Environment, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.trans_matrix = self.build_emperical_markov_matrix()

        # constructing the location visitation frequency for density epr
        loc_freq = self.env.loc_gdf.set_index("id").join(
            self.env.loc_seq_df.groupby("location_id").size().to_frame("freq")
        )
        # we assign never visited locations a small value, such that they can also be visited
        self.attr = loc_freq.fillna(0.1)["freq"].values

    def build_emperical_markov_matrix(self):
        # initialize trans matrix with 0's
        loc_size = self.env.loc_gdf["id"].max() + 1
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

        return trans_matrix.astype(np.int32)

    def explore(self, visited_loc):
        """The exploration step of the density epr model."""
        curr_loc = int(visited_loc[-1])
        curr_attr = self.attr[curr_loc]

        # delete the already visited locations
        uniq_visited_loc = set(visited_loc)
        remain_idx = np.arange(self.attr.shape[0])
        remain_idx = np.delete(remain_idx, list(uniq_visited_loc))

        # slight modification, original **2, and we changed to 1.7
        r = (self.pair_distance[curr_loc, remain_idx] / 1000) ** (1.7)
        # we also take the square root to reduce the density effect, otherwise too strong
        attr = np.power((self.attr[remain_idx] * curr_attr), 0.5).astype(float)

        # the density attraction + inverse distance
        attr = np.divide(attr, r, out=np.zeros_like(attr), where=r != 0)

        # norm
        attr = attr / attr.sum()
        selected_loc = np.random.choice(attr.shape[0], p=attr)
        return remain_idx[selected_loc]

    def pref_return(self, visited_loc):
        # not able to return to the current location
        visited_loc = np.array(visited_loc)
        curr_loc = visited_loc[-1]

        # delete the current location from the sequence
        currloc_idx = np.where(visited_loc == curr_loc)[0]
        # ensure the deleted sequence contain value
        if len(currloc_idx) != len(visited_loc):
            visited_loc = np.delete(visited_loc, currloc_idx)

        # get the transition p from emperical matrix
        curr_trans_p = self.trans_matrix[curr_loc, :]
        curr_trans_p = curr_trans_p[np.array(visited_loc)]

        # equal p if no prior knowledge
        if curr_trans_p.sum() == 0:
            curr_trans_p = np.ones_like(curr_trans_p) / len(curr_trans_p)
        else:
            curr_trans_p = curr_trans_p / curr_trans_p.sum()

        # choose next location according to emperical p
        next_loc = np.random.choice(visited_loc, p=curr_trans_p)
        # update
        self.trans_matrix[curr_loc, next_loc] += 1

        return next_loc


def post_process(traj, loc_gdf):
    def _get_result_user_df(user_seq):
        user_seq_df = pd.DataFrame(user_seq["loc_seq"], columns=["id"])
        user_seq_df["duration"] = user_seq["dur_seq"]
        user_seq_df["ori_user_id"] = user_seq["user"]

        return user_seq_df

    all_ls = []
    for i in range(len(traj)):
        user_df = _get_result_user_df(traj[i])
        user_df["user_id"] = i
        all_ls.append(user_df)

    all_df = pd.concat(all_ls).reset_index()

    # merge the geometries and sort according to visit sequence
    all_df = all_df.merge(loc_gdf, on="id", how="left", sort=False).sort_values(["user_id", "index"])

    # transfer to geodataframe
    all_gdf = gpd.GeoDataFrame(all_df, geometry="geometry", crs="EPSG:4326")

    # Final cleaning
    all_gdf = all_gdf.rename(columns={"id": "location_id", "index": "sequence"})

    return all_gdf
