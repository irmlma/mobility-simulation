import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import powerlaw
from tqdm import tqdm
import os
from pathlib import Path
import datetime

from joblib import Parallel, delayed
import contextlib
import joblib
import yaml


# pairwise distance
from scipy.spatial.distance import pdist, squareform


class Environment:
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))

        # print(self.config)

    def simulate(self):
        pass

    def simulate_agent(self):
        pass

    def simulate_agent_step(self):
        pass

    def get_wait_time(self):
        """Wait time (duration) distribution. Emperically determined from data."""
        return np.random.lognormal(self.config["wait"]["mu"], self.config["wait"]["sigma"], 1)[0]

    def get_jump(self):
        """Jump length distribution. Emperically determined from data."""
        return np.random.lognormal(self.config["jump"]["mu"], self.config["jump"]["sigma"], 1)[0]

    def get_rho(self):
        """This is learned from the emperical dataset"""
        return np.random.normal(self.config["rho"]["mu"], self.config["rho"]["sigma"], 1)[0]

    def get_gamma(self):
        """This is learned from the emperical dataset"""
        return np.random.normal(self.config["gamma"]["mu"], self.config["gamma"]["sigma"], 1)[0]


def get_initUser(user_df):
    """Randomly choose one user from the emperical dataset.

    Only useful in combination with deterministic get_initLoc()
    """
    return np.random.choice(user_df["user_id"].unique())


def get_initLoc(home_locs, user):
    """The initialization step of the epr model.

    Currently we choose the top1 visted "home" location as the start. This is deterministic.

    potential problem:
    1. as the initial location is the most important location, our synthesized dataset would not include much variety.

    Potential idea:
    1. randomly choose one location from user's top5 visited location.
    2. randomly choose one location from the location set

    Parameters
    ----------
    home_locs: the df that contains home_location and user pairs
    user: the unique user id

    Returns
    -------
    the id of the home location of the user_id
    """
    candidate = home_locs.loc[home_locs["user_id"] == user, "location_id"].values
    return int(np.random.choice(candidate))


# helper function


def load_data(data_dir):
    # data_dir = os.path.join(data_dir, "input")
    # home locs is the starting point of each user's location generation
    # home_locs_df = pd.read_csv(os.path.join(data_dir, "home_locs.csv"))
    home_locs_df = pd.read_csv(os.path.join(data_dir, "top5_locs.csv"))

    # all_sp is all the possible sp sequences, for getting the visitation frequency
    all_sp = pd.read_csv(os.path.join(data_dir, "sp.csv"), index_col="id")
    sp_p = all_sp["location_id"].value_counts().sort_index().values

    # all_locs is all the possible locations
    all_locs = pd.read_csv(os.path.join(data_dir, "locs.csv"), index_col="id")

    # for now we do not consider the user_ids and the extent
    all_locs = all_locs[~all_locs.index.duplicated(keep="first")]
    all_locs = all_locs[["center"]]
    all_locs["center"] = all_locs["center"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="center", crs="EPSG:4326")

    # the projection of Switzerland: for accurately determine the distance between two locations
    all_locs = all_locs.to_crs("EPSG:2056")
    all_locs.index = all_locs.index.astype(int)

    # precalculate distance
    coord_list = [[x, y] for x, y in zip(all_locs["center"].x, all_locs["center"].y)]
    Y = pdist(coord_list, "euclidean")
    pair_distance = squareform(Y)

    return sp_p, all_locs, home_locs_df, pair_distance, all_sp


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def base_model(generator_func, user_arr, population=10, sequence_length=10, n_jobs=-1, **kwargs):
    with tqdm_joblib(tqdm(desc="My calculation", total=population)) as _:
        results = Parallel(n_jobs=n_jobs)(
            delayed(generator_func)(sequence_length=sequence_length, user=user, **kwargs) for user in user_arr
        )

    syn_data = {}
    for i in range(len(results)):
        syn_data[i] = {}
        syn_data[i]["user"] = results[i][2]
        syn_data[i]["location_seq"] = results[i][0]
        syn_data[i]["duration_seq"] = results[i][1]

    return syn_data


def post_process(syn_data, data_dir, file_name, all_locs):
    all_ls = []
    for i in range(len(syn_data)):
        user_df = _get_result_user_df(syn_data[i], all_locs)
        user_df["user_id"] = i
        all_ls.append(user_df)

    all_df = pd.concat(all_ls).reset_index()

    all_gdf = gpd.GeoDataFrame(all_df, geometry="center", crs="EPSG:2056")
    all_gdf = all_gdf.to_crs("EPSG:4326")

    generated_dir = Path(os.path.join(data_dir, "generated"))
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    # all_gdf.to_file("data/out/generated_locs.shp")
    all_gdf.to_csv(os.path.join(generated_dir, file_name), index=False)

    # transfer to nn format
    start_time = datetime.datetime(2016, 11, 23, hour=8)
    syn_time_date = all_gdf.groupby("user_id").apply(_transfer_time, start_time)
    syn_time_date = syn_time_date.groupby("user_id").apply(_get_time_info)

    syn_time_date.rename(columns={"index": "order", "id": "location_id"}, inplace=True)
    syn_time_date.drop(columns={"center", "proj_time"}, inplace=True)
    syn_time_date.index.name = "id"
    syn_time_date.reset_index(inplace=True)

    nn_dir = Path(os.path.join(data_dir, "nn"))
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)
    syn_time_date.to_csv(os.path.join(nn_dir, file_name), index=False)
    print("Final user size: ", syn_time_date["user_id"].unique().shape[0])


def _get_result_user_df(user_sequence, all_locs):
    user_sequence_df = pd.DataFrame(user_sequence["location_seq"], columns=["id"])
    user_sequence_df["duration"] = user_sequence["duration_seq"]
    user_sequence_df["ori_user_id"] = user_sequence["user"]
    geom_df = (
        user_sequence_df.reset_index()
        .merge(all_locs.reset_index(), on="id", sort=False)
        .sort_values(by="index")
        .reset_index(drop=True)
        .drop(columns="index")
    )
    return geom_df


def _transfer_time(df, start_time):
    duration_arr = df["duration"].to_list()[:-1]
    duration_arr.insert(0, 0)
    timedelta_arr = np.array([datetime.timedelta(hours=i) for i in np.cumsum(duration_arr)])

    df["proj_time"] = timedelta_arr + start_time

    return df


def _get_time_info(df):
    min_day = pd.to_datetime(df["proj_time"].min().date())

    df["start_day"] = (df["proj_time"] - min_day).dt.days

    df["start_min"] = df["proj_time"].dt.hour * 60 + df["proj_time"].dt.minute

    df["weekday"] = df["proj_time"].dt.weekday

    df["duration"] = (df["duration"] * 60).round()
    return df


def _filter_infrequent_group(df, min_count=5):
    """filter infrequent locations"""
    value_counts = df["location_id"].value_counts()
    valid = value_counts[value_counts > min_count]

    return df.loc[df["location_id"].isin(valid.index)].copy()


def _filter_infrequent_user(df, min_count=5):
    """filter infrequent locations"""
    value_counts = df["location_id"].value_counts()
    valid = value_counts[value_counts > min_count]

    return df.loc[df["location_id"].isin(valid.index)].copy()
