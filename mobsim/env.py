import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import os

import powerlaw

import yaml


class Environment:
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))

        # location visit sequence
        self.loc_seq_df = pd.read_csv(os.path.join(self.config["data_dir"], self.config["loc_seq_file"]))

        # top location visited by each user (empirical)
        # for determining the start location for simulation, we choose the top 5 as possible locations
        self.top_user_loc_df = (
            self.loc_seq_df.groupby(["user_id", "location_id"], as_index=False)
            .size()
            .sort_values(by="size", ascending=False)
            .groupby(["user_id"])
            .head(5)
        )

        # location with geom
        loc_df = pd.read_csv(os.path.join(self.config["data_dir"], self.config["locs_file"]))
        loc_df["geometry"] = loc_df["geometry"].apply(wkt.loads)
        self.loc_gdf = gpd.GeoDataFrame(loc_df, geometry="geometry", crs="EPSG:4326")

    def get_wait_time(self):
        """Wait time (duration) distribution. Emperically determined from data."""
        if self.config["wait"]["type"] == "lognormal":
            return np.random.lognormal(self.config["wait"]["mu"], self.config["wait"]["sigma"], 1)[0]
        elif self.config["wait"]["type"] == "powerlaw":
            return powerlaw.Power_Law(parameters=[self.config["wait"]["alpha"]]).generate_random(n=1)[0]
        elif self.config["wait"]["type"] == "truncpowerlaw":
            return powerlaw.Truncated_Power_Law(
                parameters=[self.config["wait"]["alpha"], self.config["wait"]["Lambda"]]
            ).generate_random(n=1)[0]
        else:
            raise AttributeError(
                f"Distribution for wait time not supported. Please check the input arguement. We only support 'lognormal', 'powerlaw' or 'truncpowerlaw'. You passed {self.config['wait']['type']}"
            )

    def get_jump(self):
        """Jump length distribution. Emperically determined from data."""
        if self.config["jump"]["type"] == "lognormal":
            return np.random.lognormal(self.config["jump"]["mu"], self.config["jump"]["sigma"], 1)[0]
        elif self.config["jump"]["type"] == "powerlaw":
            return powerlaw.Power_Law(parameters=[self.config["jump"]["alpha"]]).generate_random(n=1)[0]
        elif self.config["jump"]["type"] == "truncpowerlaw":
            return powerlaw.Truncated_Power_Law(
                parameters=[self.config["jump"]["alpha"], self.config["jump"]["Lambda"]]
            ).generate_random(n=1)[0]
        else:
            raise AttributeError(
                f"Distribution for jump lengths not supported. Please check the input arguement. We only support 'lognormal', 'powerlaw' or 'truncpowerlaw'. You passed {self.config['jump']['type']}"
            )

    def get_rho(self):
        """This is learned from the emperical dataset"""
        return np.random.normal(self.config["rho"]["mu"], self.config["rho"]["sigma"], 1)[0]

    def get_gamma(self):
        """This is learned from the emperical dataset"""
        return np.random.normal(self.config["gamma"]["mu"], self.config["gamma"]["sigma"], 1)[0]
