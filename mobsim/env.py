import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import os

import yaml


class Environment:
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))

        # top location visited by each user (empirical)
        # for determining the start location for simulation
        self.top_user_loc_df = pd.read_csv(os.path.join(self.config["data_dir"], "top5_locs.csv"))

        # location visit sequence
        self.loc_seq_df = pd.read_csv(os.path.join(self.config["data_dir"], "loc_seq.csv"))

        # location with geom and visitation freqency
        loc_df = pd.read_csv(os.path.join(self.config["data_dir"], "loc_freq.csv"))
        loc_df["center"] = loc_df["center"].apply(wkt.loads)
        self.loc_gdf = gpd.GeoDataFrame(loc_df, geometry="center", crs="EPSG:2056")

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
