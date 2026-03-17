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
        
        # PCA transformation applied to normalized laps per driver
        self.reduced_dict = {}
        for driver in self.drivers:
            self.reduced_dict[driver] = []

        self._load()
        self._get_min_max()
        self._reindex()
        self._normalize()
        self._average_speed_check()
        self.pca()

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

    # CHUNKS OF 5 TO 10, 0-5, 0-10
    # Finds the global min/max for each column across all drivers and laps, used for normalization
    def _get_min_max(self):
        for driver in self.drivers:
            for df in self.df_dict[driver]:
                for col in self.norm_columns:
                    if df[col].count() == 0:
                        continue
                    self.max_dict[col] = max(self.max_dict.get(col, -np.inf), np.percentile(df[col], 98))
                    self.min_dict[col] = min(self.min_dict.get(col, np.inf), np.percentile(df[col], 2))

    def _get_min_max_driver_lap(self, driver, laps_min, laps_max):
        min_dict, max_dict = {}, {}

        if(len(self.df_dict[driver]) < laps_max):
            laps_max = len(self.df_dict[driver])
            laps_min = laps_max - 5

        for df in self.df_dict[driver][laps_min:laps_max]:
            for col in self.norm_columns:
                if df[col].count() == 0:
                    continue
                max_dict[col] = max(max_dict.get(col, -np.inf), np.percentile(df[col], 98))
                min_dict[col] = min(min_dict.get(col, np.inf), np.percentile(df[col], 2))
        
        return min_dict, max_dict

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
        for driver in self.drivers:
            for lap_num, df in enumerate(self.df_dict[driver], start=1):
                df = self._reindex_df_operations(df)

                if df.isna().sum().sum() > 1:
                    print(f"  Dropping {driver} lap {lap_num} — NaN values detected")
                    continue

                df['gear'] = df['gear'].fillna(0).round().astype(int)
                self.interp_dict[driver].append(df)

    # Applies min-max normalization to each column in norm_columns
    def _normalize(self):
        print("Normalizing...")
        for driver in self.drivers:
            print(f"  Normalizing {driver}...")
            prev_lap_increment = 0
            min_dict, max_dict = {}, {}
            for lap_num, df in enumerate(self.interp_dict[driver], start=1):
                lap_increment = lap_num + (5 - lap_num % 5)

                if lap_increment != prev_lap_increment:
                    min_dict, max_dict = self._get_min_max_driver_lap(driver, lap_increment - 5, lap_increment)
                    prev_lap_increment = lap_increment
                
                for col in self.norm_columns:
                    df[col] = (df[col] - min_dict[col]) / (max_dict[col] - min_dict[col])
                    np.clip(df[col], 0, 1)

    def _average_speed_check(self):
        print("Checking speed thresholds...")

        for driver in self.drivers:
            filtered_laps = []

            for lap_num, df in enumerate(self.interp_dict[driver], start=1):
                avg_speed = np.mean(df['speed'])

                ## These arbitrary lap nums saw the highest proportion of slow laps among drivers
                ## Safer to remove them for all drivers to not skew clusters
                if(avg_speed <= 0.6 or lap_num == 1 or lap_num >= 66):
                    print(f"  Dropping {driver} lap {lap_num} — too slow or filtered out lap number (1, 66+)")
                else:
                    filtered_laps.append(df)
            
            self.interp_dict[driver] = filtered_laps

    def pca(self, n_components=0.95):
        all_points = []

        for driver in self.drivers:
            for lap_df in self.interp_dict[driver]:
                all_points.append(lap_df[self.norm_columns].values)

        X = np.vstack(all_points)

        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)

        print("\nApplying PCA...")
        print(f"  Components created: {pca.n_components_}")
        print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

        for driver in self.drivers:
            for lap_df in self.interp_dict[driver]:
                X_lap = lap_df[self.norm_columns].values
                X_reduced = pca.transform(X_lap)
                self.reduced_dict[driver].append(X_reduced)

if __name__ == '__main__':
    race = RaceData('Canadian_Grand_Prix')

    pca, X_reduced = race.pca()
