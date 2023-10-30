import os
import datetime
import pandas as pd
import numpy as np

import argparse
import pickle as pickle
from pathlib import Path

import powerlaw
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

from joblib import Parallel, delayed
import multiprocessing

import geopandas as gpd
from shapely import wkt

from tqdm import tqdm

# trackintel
from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps
from trackintel.geogr.point_distances import haversine_dist
from trackintel.preprocessing.triplegs import generate_trips
import trackintel as ti


def get_dataset(raw_data_dir, bound_file, epsilon=20, num_samples=1):
    """Construct the raw staypoint with location id dataset from GC data."""
    # read file storage
    ## read and change name to trackintel format
    sp = pd.read_csv(os.path.join(raw_data_dir, "stps.csv"))
    tpls = pd.read_csv(os.path.join(raw_data_dir, "tpls.csv"))
    # initial cleaning
    sp.rename(columns={"activity": "is_activity"}, inplace=True)

    sp = _preprocess_to_ti(sp)
    tpls = _preprocess_to_ti(tpls)

    sp = ti.io.from_geopandas.read_staypoints_gpd(sp, geom_col="geom", crs="EPSG:4326", tz="utc")
    tpls = ti.io.from_geopandas.read_triplegs_gpd(tpls, geom_col="geom", crs="EPSG:4326", tz="utc")

    # ensure the timeline of sp and tpls does not overlap
    sp, tpls = _filter_duplicates(sp.copy().reset_index(), tpls.reset_index())

    ## select valid user
    quality_dir = os.path.join(".", "data", "quality")
    quality_file = os.path.join(quality_dir, "gc_slide_filtered.csv")
    if Path(quality_file).is_file():
        valid_users = pd.read_csv(quality_file)["user_id"].values
    else:
        if not os.path.exists(quality_dir):
            os.makedirs(quality_dir)
        # the trackintel trip generation
        sp, tpls, trips = generate_trips(sp, tpls, add_geometry=False, gap_threshold=15)
        quality_filter = {"day_filter": 300, "window_size": 10, "min_thres": 0.6, "mean_thres": 0.7}
        valid_users = _calculate_user_quality(sp.copy(), trips.copy(), quality_file, quality_filter)

    sp = sp.loc[sp["user_id"].isin(valid_users)]

    ## select only switzerland records
    boundary_gdf = gpd.read_file(bound_file)
    print("Before spatial filtering: ", sp.shape[0])
    sp = _filter_within_bound(sp, boundary_gdf)
    print("After spatial filtering: ", sp.shape[0])

    # filter activity staypoints
    sp = sp.loc[sp["is_activity"] == True]

    # generate locations
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=epsilon, num_samples=num_samples, distance_metric="haversine", agg_level="dataset", n_jobs=-1
    )
    # filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print("After filter non-location staypoints: ", sp.shape[0])

    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    # merge staypoints
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]), max_time_gap="1min", agg={"location_id": "first"}
    )
    print("After staypoints merging: ", sp_merged.shape[0])
    # recalculate staypoint duration
    sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60

    # reindex sp and locs
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    sp_merged["location_id"] = enc.fit_transform(sp_merged["location_id"].values.reshape(-1, 1))
    locs_index = locs.reset_index()
    locs_index["id"] = enc.transform(locs_index["id"].values.reshape(-1, 1))
    locs = locs_index.loc[locs_index["id"] != -1]

    # cleaning
    locs.drop(columns={"user_id", "extent"}, inplace=True)
    locs.rename(columns={"center": "geometry"}, inplace=True)
    locs = locs.drop_duplicates(subset="id").sort_values(by="id")
    locs["id"] = locs["id"].astype(int)

    sp = sp_merged.sort_values(by=["user_id", "started_at"])[["location_id", "user_id"]]
    sp["location_id"] = sp["location_id"].astype(int)

    # save results
    locs.to_csv("./data/input/locs.csv", index=False)
    sp.to_csv("./data/input/locs_seq.csv", index=False)

    # get the paraemetrs for mobility generation models
    get_epr_distributions(sp_merged, locs)


def get_epr_distributions(sp, locs):
    def getAIC(fit, empr):
        aics = []

        aics.append(-2 * np.sum(fit.truncated_power_law.loglikelihoods(empr)) + 4)
        aics.append(-2 * np.sum(fit.power_law.loglikelihoods(empr)) + 2)
        aics.append(-2 * np.sum(fit.lognormal.loglikelihoods(empr)) + 4)

        aics = aics - np.min(aics)

        down = np.sum([np.exp(-aic / 2) for aic in aics])

        res = {}
        res["truncated_power_law"] = np.exp(-aics[0] / 2) / down
        res["power_law"] = np.exp(-aics[1] / 2) / down
        res["lognormal"] = np.exp(-aics[2] / 2) / down

        return res

    def get_jump_length(gdf):
        geom_arr = gdf.geometry.values

        res_ls = []
        for i in range(1, len(geom_arr)):
            res_ls.append(haversine_dist(geom_arr[i - 1].x, geom_arr[i - 1].y, geom_arr[i].x, geom_arr[i].y)[0])
        return res_ls

    sp = sp.merge(locs.rename(columns={"id": "location_id"}), how="left", on="location_id")

    jump_length = gpd.GeoDataFrame(sp, geometry="geometry").groupby("user_id").apply(get_jump_length)
    flat_jump_length = np.array([item for sublist in jump_length.to_list() for item in sublist])
    flat_jump_length = flat_jump_length[flat_jump_length > 5]

    fit = powerlaw.Fit(flat_jump_length, xmin=flat_jump_length.min(), xmin_distribution="lognormal")
    print("AIC criteria:", getAIC(fit, flat_jump_length))
    print(f"Lognormal: parameter1: {fit.lognormal.parameter1:.2f}\t parameter2: {fit.lognormal.parameter2:.2f}")
    print(
        f"Truncated power law: parameter1: {fit.truncated_power_law.parameter1:.2f}\t parameter2: {fit.truncated_power_law.parameter2:.2f}"
    )
    print(f"Power law: alpha: {fit.power_law.alpha:.2f}")

    duration_hour = ((sp["finished_at"] - sp["started_at"]).dt.total_seconds() / 3600).values
    duration_hour = duration_hour[duration_hour > 0.2]

    fit = powerlaw.Fit(duration_hour, xmin=duration_hour.min(), xmin_distribution="lognormal")
    print("AIC criteria:", getAIC(fit, duration_hour))
    print(f"Lognormal: parameter1: {fit.lognormal.parameter1:.2f}\t parameter2: {fit.lognormal.parameter2:.2f}")
    print(
        f"Truncated power law: parameter1: {fit.truncated_power_law.parameter1:.2f}\t parameter2: {fit.truncated_power_law.parameter2:.2f}"
    )
    print(f"Power law: alpha: {fit.power_law.alpha:.2f}")

    def fit_powerlaw(df):
        loc = df["location_id"].values

        unique_count_ls = []
        for i in range(loc.shape[0]):
            unique_count_ls.append(len(np.unique(loc[: i + 1])))

        # big S
        unique_count_arr = np.array(unique_count_ls)

        # small n
        steps = np.arange(unique_count_arr.shape[0]) + 1

        logy = np.log(unique_count_arr)
        logx = np.log(steps)
        # print(logy, logx)
        reg = LinearRegression().fit(logx.reshape(-1, 1), logy)

        r = 1 / reg.coef_ - 1
        p = np.exp((reg.intercept_ - np.log(1 + r)) * (1 + r))

        return pd.Series([r[0], p[0]], index=["r", "p"])

    explor_para = sp.groupby("user_id").apply(fit_powerlaw)

    print("Normal distribution parameters for gamma:", norm.fit(explor_para.r))
    print("Normal distribution parameters for rho:", norm.fit(explor_para.p))


def _preprocess_to_ti(df):
    """Change dataframe to trackintel compatible format"""
    df.rename(
        columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at", "dur_s": "duration"}, inplace=True
    )

    tqdm.pandas(desc="Loading Geometry")
    df["geom"] = df["geom"].progress_apply(wkt.loads)

    return df


def _filter_duplicates(sp, tpls):
    # merge trips and staypoints
    sp["type"] = "sp"
    tpls["type"] = "tpl"
    df_all = pd.merge(sp, tpls, how="outer")

    def alter_diff(df):
        df.sort_values(by="started_at", inplace=True)
        df["diff"] = pd.NA
        df["st_next"] = pd.NA

        diff = df["started_at"].iloc[1:].reset_index(drop=True) - df["finished_at"].iloc[:-1].reset_index(drop=True)
        df["diff"].iloc[:-1] = diff.dt.total_seconds()
        df["st_next"].iloc[:-1] = df["started_at"].iloc[1:].reset_index(drop=True)

        df.loc[df["diff"] < 0, "finished_at"] = df.loc[df["diff"] < 0, "st_next"]

        df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
        df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()

        # print(df.loc[df["diff"] < 0])
        df.drop(columns=["diff", "st_next"], inplace=True)
        df.drop(index=df[df["duration"] <= 0].index, inplace=True)

        return df

    df_all = df_all.groupby("user_id", as_index=False).apply(alter_diff)
    sp = df_all.loc[df_all["type"] == "sp"].drop(columns=["type"])
    tpls = df_all.loc[df_all["type"] == "tpl"].drop(columns=["type"])

    sp = sp[["id", "user_id", "started_at", "finished_at", "geom", "duration", "is_activity"]]
    tpls = tpls[["id", "user_id", "started_at", "finished_at", "geom", "length_m", "duration", "mode"]]

    return sp.set_index("id"), tpls.set_index("id")


def _calculate_user_quality(sp, trips, file_path, quality_filter):
    trips["started_at"] = pd.to_datetime(trips["started_at"]).dt.tz_localize(None)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"]).dt.tz_localize(None)
    sp["started_at"] = pd.to_datetime(sp["started_at"]).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"]).dt.tz_localize(None)

    # merge trips and staypoints for getting the tracking quality
    print("starting merge", sp.shape, trips.shape)
    sp["type"] = "sp"
    trips["type"] = "tpl"
    df_all = pd.concat([sp, trips])
    df_all = _split_overlaps(df_all, granularity="day")
    df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()
    print("finished merge", df_all.shape)
    print("*" * 50)

    # remove tracking records after the tracking end
    end_period = datetime.datetime(2017, 12, 26)
    df_all = df_all.loc[df_all["finished_at"] < end_period]

    # get quality
    total_quality = temporal_tracking_quality(df_all, granularity="all")
    # get tracking days
    total_quality["days"] = (
        df_all.groupby("user_id").apply(lambda x: (x["finished_at"].max() - x["started_at"].min()).days).values
    )
    # filter based on days
    user_filter_day = (
        total_quality.loc[(total_quality["days"] > quality_filter["day_filter"])]
        .reset_index(drop=True)["user_id"]
        .unique()
    )

    def get_tracking_quality(df, window_size):
        weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
        start_date = df["started_at"].min().date()

        quality_list = []
        # construct the sliding week gdf
        for i in range(0, weeks - window_size + 1):
            curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
            curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

            # the total df for this time window
            cAll_gdf = df.loc[(df["started_at"] >= curr_start) & (df["finished_at"] < curr_end)]
            if cAll_gdf.shape[0] == 0:
                continue
            total_sec = (curr_end - curr_start).total_seconds()

            quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
        ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
        ret["user_id"] = df["user_id"].unique()[0]
        return ret

    sliding_quality = (
        df_all.groupby("user_id")
        .apply(get_tracking_quality, window_size=quality_filter["window_size"])
        .reset_index(drop=True)
    )

    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]

    if "min_thres" in quality_filter:

        def filter_user(df, min_thres, mean_thres):
            consider = df.loc[df["quality"] != 0]
            if (consider["quality"].min() > min_thres) and (consider["quality"].mean() > mean_thres):
                return df

        # filter based on quanlity
        filter_after_day = (
            filter_after_day.groupby("user_id")
            .apply(filter_user, min_thres=quality_filter["min_thres"], mean_thres=quality_filter["mean_thres"])
            .reset_index(drop=True)
            .dropna()
        )

    filter_after_user_quality = filter_after_day.groupby("user_id", as_index=False)["quality"].mean()

    print("final selected user", filter_after_user_quality.shape[0])
    filter_after_user_quality.to_csv(file_path, index=False)
    return filter_after_user_quality["user_id"].values


def _filter_within_bound(stps, bound):
    """Spatial filtering of staypoints."""
    # save a copy of the original projection
    init_crs = stps.crs
    # project to projected system
    stps = stps.to_crs(bound.crs)

    ## parallel for speeding up
    stps["within"] = _apply_parallel(stps["geom"], _apply_extract, bound)
    sp_swiss = stps[stps["within"] == True].copy()
    sp_swiss.drop(columns=["within"], inplace=True)

    return sp_swiss.to_crs(init_crs)


def _apply_extract(df, swissBound):
    """The func for _apply_parallel: judge whether inside a shp."""
    tqdm.pandas(desc="pandas bar")
    shp = swissBound["geometry"].to_numpy()[0]
    return df.progress_apply(lambda x: shp.contains(x))


def _apply_parallel(df, func, other, n=-1):
    """parallel apply for spending up."""
    if n is None:
        n = -1
    dflength = len(df)
    cpunum = multiprocessing.cpu_count()
    if dflength < cpunum:
        spnum = dflength
    if n < 0:
        spnum = cpunum + n + 1
    else:
        spnum = n or 1

    sp = list(range(dflength)[:: int(dflength / spnum + 0.5)])
    sp.append(dflength)
    slice_gen = (slice(*idx) for idx in zip(sp[:-1], sp[1:]))
    results = Parallel(n_jobs=n, verbose=0)(delayed(func)(df.iloc[slc], other) for slc in slice_gen)
    return pd.concat(results)


if __name__ == "__main__":
    # read file storage

    parser = argparse.ArgumentParser()
    parser.add_argument("epsilon", type=int, nargs="?", help="epsilon for dbscan to detect locations", default=20)
    parser.add_argument("num_pts", type=int, nargs="?", help="num_samples for dbscan to detect locations", default=1)
    parser.add_argument(
        "raw_data_dir", type=str, nargs="?", help="dir for data storage", default="D:/SBBGC1/raw_re_trip"
    )
    parser.add_argument(
        "bound_file",
        type=str,
        nargs="?",
        help="filename of swiss boundary",
        default="./data/bound/swiss_1903+.shp",
    )
    args = parser.parse_args()

    get_dataset(epsilon=args.epsilon, num_pts=args.num_pts, raw_data_dir=args.raw_data_dir, bound_file=args.bound_file)
