import json
import numpy as np
import pandas as pd
from query_db import TelemetryDatabase
from sklearn.decomposition import PCA

CACHE_FILE = 'cache/race_metadata_cache.json'

# Load race schedule and driver lap counts from the JSON cachea
with open(CACHE_FILE) as f:
    _cache = json.load(f)

RACES = _cache['races']
RACE_NAMES = list(RACES.keys())

# Maps race name -> {driver: lap_count}
DRIVER_LAPS = {}
for race in RACE_NAMES:
    DRIVER_LAPS[race] = RACES[race]['driver_laps']

# Maps race name -> [driver codes]
DRIVERS = {}
for race in RACE_NAMES:
    DRIVERS[race] = list(RACES[race]['driver_laps'].keys())

# All columns fetched from the database
TEL_COLUMNS = ['rel_distance', 'time', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'drs', 'speed', 'acc_x', 'acc_y', 'acc_z']

# Default columns to normalize. Can be overridden by the user at init
NORM_TEL_COLUMNS = ['rpm', 'gear', 'throttle', 'speed', 'acc_x', 'acc_y', 'acc_z']

# Number of evenly spaced points each lap is interpolated to
DATAPOINTS_PER_LAP = 500


class RaceData:

    # Pass a race name to load, process, and normalize all telemetry for that race.
    # Optionally pass norm_columns to normalize a custom set of columns.
    def __init__(self, race_name, norm_columns=None):
        self.race_name = race_name
        self.drivers = DRIVERS[race_name]
        self.driver_laps = DRIVER_LAPS[race_name]
        self.db = TelemetryDatabase()
        self.max_dict = {}
        self.min_dict = {}

        self.norm_columns = NORM_TEL_COLUMNS
        if norm_columns:
            self.norm_columns = norm_columns

        # Raw laps per driver, indexed by rel_distance
        self.df_dict = {}
        for driver in self.drivers:
            self.df_dict[driver] = []

        # Interpolated and normalized laps per driver
        self.interp_dict = {}
        for driver in self.drivers:
            self.interp_dict[driver] = []

        self._load()
        self._get_min_max()
        self._reindex()
        self._normalize()

    # Returns normalized laps for a single driver, or all drivers if none specified
    def get(self, driver=None):
        if driver:
            return self.interp_dict[driver]
        return self.interp_dict

    # Fetches all laps for all drivers from the database and stores them in df_dict
    def _load(self):
        print(f"Loading data for {self.race_name}...")
        for driver in self.drivers:
            print(f"  Loading {driver}...")
            for lap in range(1, self.driver_laps[driver] + 1):
                df = self.db.fetch_driver_telemetry_by_lap(self.race_name, driver, TEL_COLUMNS, lap_num=lap)
                df.set_index('rel_distance', inplace=True)
                df['brake'] = df['brake'].astype(int)
                df['drs'] = df['drs'].astype(int)
                self.df_dict[driver].append(df)

    # Finds the global min/max for each column across all drivers and laps, used for normalization
    def _get_min_max(self):
        for driver in self.drivers:
            for df in self.df_dict[driver]:
                for col in self.norm_columns:
                    self.max_dict[col] = max(self.max_dict.get(col, -np.inf), df[col].max())
                    self.min_dict[col] = min(self.min_dict.get(col, np.inf), df[col].min())

    def _reindex_df_operations(self, df):
        uniform_index = np.linspace(0, 1, DATAPOINTS_PER_LAP)
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(df.index.union(uniform_index))
        df = df.infer_objects(copy=False)
        df = df.interpolate(method='index')
        df = df.ffill()
        df = df.bfill()
        df = df.reindex(np.linspace(0, 1, DATAPOINTS_PER_LAP))

        return df

    # Interpolates each lap onto a uniform grid of DATAPOINTS_PER_LAP points between 0 and 1
    def _reindex(self):
        print("Reindexing...")
        uniform_index = np.linspace(0, 1, DATAPOINTS_PER_LAP)
        for driver in self.drivers:
            for lap_num, df in enumerate(self.df_dict[driver], start=1):
                df = self._reindex_df_operations(df)

                if df.isna().sum().sum() > 1:
                    print(f"  Dropping {driver} lap {lap_num} — too many NaN values")
                    continue

                df['gear'] = df['gear'].fillna(0).round().astype(int)
                self.interp_dict[driver].append(df)

    # Applies min-max normalization to each column in norm_columns
    def _normalize(self):
        print("Normalizing...")
        for driver in self.drivers:
            for df in self.interp_dict[driver]:
                for col in self.norm_columns:
                    df[col] = (df[col] - self.min_dict[col]) / (self.max_dict[col] - self.min_dict[col])

    def pca(self, n_components=0.95):
        all_points = []

        for driver in self.drivers:
            for lap_df in self.interp_dict[driver]:
                all_points.append(lap_df[self.norm_columns].values)

        X = np.vstack(all_points)
        print(f"\nInput shape: {X.shape}")

        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)

        print(f"Components kept: {pca.n_components_}")
        print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        print(f"Per-component variance: {np.round(pca.explained_variance_ratio_, 3)}")
        print("Feature loadings (components x features):")
        for i, component in enumerate(pca.components_):
            print(f"\n  PC{i+1}:")
            for col, loading in zip(self.norm_columns, component):
                bar = '█' * int(abs(loading) * 20)
                sign = '+' if loading >= 0 else '-'
                print(f"    {col:<12} {sign}{abs(loading):.3f}  {bar}")
        reduced_dict = {}
        for driver in self.drivers:
            reduced_dict[driver] = []
            for lap_df in self.interp_dict[driver]:
                X_lap = lap_df[self.norm_columns].values
                X_reduced = pca.transform(X_lap)
                reduced_dict[driver].append(X_reduced)

        return pca, reduced_dict


if __name__ == '__main__':
    race = RaceData('Canadian_Grand_Prix')

    pca, X_reduced = race.pca()