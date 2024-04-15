import pandas as pd
import geopandas as gpd
import math

import numpy as np
from tqdm import tqdm

from mobsim.env import Environment

from sklearn.metrics import pairwise_distances
import shapely


class EPR:
    """Explore and preferential return model"""

    def __init__(self, env: Environment):
        self.env = env

        # precalculate distance between location pairs
        if self.env.proj_crs is not None:
            # the projection of Switzerland: for accurately determine the distance between two locations
            self.env.loc_gdf = self.env.loc_gdf.to_crs(self.env.proj_crs)
            self.pair_distance = calculate_distance_matrix(self.env.loc_gdf, n_jobs=-1, dist_metric="euclidean")
            self.env.loc_gdf = self.env.loc_gdf.to_crs("EPSG:4326")
        else:
            self.pair_distance = calculate_distance_matrix(self.env.loc_gdf, n_jobs=-1, dist_metric="haversine")

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
        loc_freq = self.env.loc_gdf.set_index("location_id").join(
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
        loc_size = self.env.loc_gdf["location_id"].max() + 1
        trans_matrix = np.zeros([loc_size, loc_size])
        # transform transition list to matrix
        tran_ls = self.env.loc_seq_df.groupby("user_id").apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x["location_id"].values, 2)
        )
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
        loc_freq = self.env.loc_gdf.set_index("location_id").join(
            self.env.loc_seq_df.groupby("location_id").size().to_frame("freq")
        )
        # we assign never visited locations a small value, such that they can also be visited
        self.attr = loc_freq.fillna(0.1)["freq"].values

    def build_emperical_markov_matrix(self):
        # initialize trans matrix with 0's
        loc_size = self.env.loc_gdf["location_id"].max() + 1
        trans_matrix = np.zeros([loc_size, loc_size])

        # transform transition list to matrix
        tran_ls = self.env.loc_seq_df.groupby("user_id").apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x["location_id"].values, 2)
        )
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
        user_seq_df = pd.DataFrame(user_seq["loc_seq"], columns=["location_id"])
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
    all_df = all_df.merge(loc_gdf, on="location_id", how="left", sort=False).sort_values(["user_id", "index"])

    # transfer to geodataframe
    all_gdf = gpd.GeoDataFrame(all_df, geometry="geometry", crs="EPSG:4326")

    # Final cleaning
    all_gdf = all_gdf.rename(columns={"index": "sequence"})

    return all_gdf


def calculate_distance_matrix(X, Y=None, dist_metric="haversine", n_jobs=None, **kwds):
    if dist_metric == "haversine":
        # curry our haversine distance
        def haversine_curry(a, b, **_):
            return point_haversine_dist(*a, *b, float_flag=True)

        dist_metric = haversine_curry
    X = shapely.get_coordinates(X.geometry)
    Y = shapely.get_coordinates(Y.geometry) if Y is not None else X
    return pairwise_distances(X, Y, metric=dist_metric, n_jobs=n_jobs, **kwds)


def point_haversine_dist(lon_1, lat_1, lon_2, lat_2, r=6371000, float_flag=False):
    """
    Compute the great circle or haversine distance between two coordinates in WGS84.

    Serialized version of the haversine distance.

    Parameters
    ----------
    lon_1 : float or numpy.array of shape (-1,)
        The longitude of the first point.

    lat_1 : float or numpy.array of shape (-1,)
        The latitude of the first point.

    lon_2 : float or numpy.array of shape (-1,)
        The longitude of the second point.

    lat_2 : float or numpy.array of shape (-1,)
        The latitude of the second point.

    r     : float
        Radius of the reference sphere for the calculation.
        The average Earth radius is 6'371'000 m.

    float_flag : bool, default False
        Optimization flag. Set to True if you are sure that you are only using floats as args.

    Returns
    -------
    float or numpy.array
        An approximation of the distance between two points in WGS84 given in meters.

    Examples
    --------
    >>> point_haversine_dist(8.5, 47.3, 8.7, 47.2)
    18749.056277719905

    References
    ----------
    https://en.wikipedia.org/wiki/Haversine_formula
    https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
    """
    if float_flag:
        lon_1 = math.radians(lon_1)
        lat_1 = math.radians(lat_1)
        lon_2 = math.radians(lon_2)
        lat_2 = math.radians(lat_2)

        cos_lat2 = math.cos(lat_2)
        cos_lat1 = math.cos(lat_1)
        cos_lat_d = math.cos(lat_1 - lat_2)
        cos_lon_d = math.cos(lon_1 - lon_2)

        return r * math.acos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

    lon_1 = np.deg2rad(lon_1).ravel()
    lat_1 = np.deg2rad(lat_1).ravel()
    lon_2 = np.deg2rad(lon_2).ravel()
    lat_2 = np.deg2rad(lat_2).ravel()

    cos_lat1 = np.cos(lat_1)
    cos_lat2 = np.cos(lat_2)
    cos_lat_d = np.cos(lat_1 - lat_2)
    cos_lon_d = np.cos(lon_1 - lon_2)

    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))
