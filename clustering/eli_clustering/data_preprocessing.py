from data_ingestion.query_db import query_db
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np

## Variable declaration. Can be done with fastf1, but this is far easier at the moment
telemetry_columns = ['rel_distance', 'time', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'speed', 'acc_x', 'acc_y', 'acc_z']
normalized_telemetry_columns = ['rpm', 'gear', 'throttle', '', 'speed', 'acc_x', 'acc_y', 'acc_z']
canadian_gp_drivers = ['RUS', 'VER', 'ANT', 'PIA', 'LEC', 'HAM', 'ALO', 'HUL', 'OCO', 'SAI', 'BEA', 'TSU', 'COL', 'BOR', 'GAS', 'HAD', 'STR', 'NOR', 'LAW', 'ALB']
canadian_gp_driver_laps = {'RUS': 70, 'VER': 70, 'ANT': 70, 'PIA': 70, 'LEC': 70, 'HAM': 70, 'ALO': 70, 'HUL': 70, 'OCO': 69, 'SAI': 69, 'BEA': 69, 'TSU': 69, 'COL': 69, 'BOR': 69, 'GAS': 69, 'HAD': 69, 'STR': 69, 'NOR': 66, 'LAW': 53, 'ALB': 46}
canadian_gp_drivers_df_dict = {'RUS': [], 'VER': [], 'ANT': [], 'PIA': [], 'LEC': [], 'HAM': [], 'ALO': [], 'HUL': [], 'OCO': [], 'SAI': [], 'BEA': [], 'TSU': [], 'COL': [], 'BOR': [], 'GAS': [], 'HAD': [], 'STR': [], 'NOR': [], 'LAW': [], 'ALB': []}

datapoints_per_lap = 50000
db = query_db()
max_dict = {}
min_dict = {}


## Function used for querying the database and storing the dataframes locally. EXTREMELY SLOW AT THE MOMENT
def query_and_df_creation():

    for driver in canadian_gp_drivers:
        lap_count = canadian_gp_driver_laps[driver]
        for lap in range(1, lap_count + 1):
            driver_df = db.fetch_driver_telemetry_by_lap('CAN', driver, telemetry_columns, lap_num=lap)
            driver_df.set_index('rel_distance', inplace=True)
            driver_df['brake'] = driver_df['brake'].astype(int)

            canadian_gp_drivers_df_dict[driver].append(driver_df)
            driver_df.to_pickle(f'clustering/eli_clustering/pandas_df/{driver}{lap}')

## Finds the maximum / minimum values of all metrics for normalization purposes. Looks at every driver, every lap
def get_max_values():
    for driver in canadian_gp_drivers_df_dict.keys():

        for i in range(canadian_gp_driver_laps[driver]):
            driver_df = canadian_gp_drivers_df_dict[driver][i]
            
            for column in normalized_telemetry_columns:
                col_max = driver_df[column].max()
                col_min = driver_df[column].min()
                max_dict[column] = max(max_dict.get(column, -np.inf), col_max)
                min_dict[column] = min(min_dict.get(column, np.inf), col_min)

## Repopulates the dataframes dict if they are stored locally
def pickle_df_repop():
    for driver in canadian_gp_drivers:
        for i in range(1, canadian_gp_driver_laps[driver] + 1):
            temp_driver_df = pd.read_pickle(f'clustering/eli_clustering/pandas_df/{driver}{i}')
            canadian_gp_drivers_df_dict[driver].append(temp_driver_df)

## Normalization of the data / interpolation is performed
def interpolate():
    uniform_index = np.linspace(0, 1, datapoints_per_lap)
    # for driver in driver_df_dict.keys():
    driver = 'RUS'
    driver_df = canadian_gp_drivers_df_dict[driver][0]
    driver_df = driver_df[~driver_df.index.duplicated(keep='first')]
    print(driver_df)

    # Combine original index with uniform grid
    combined_index = driver_df.index.union(uniform_index)

    # Reindex to combined index (inserts NaNs at new grid points)
    driver_df = driver_df.reindex(combined_index)

    # Interpolate to fill in the NaNs
    driver_df = driver_df.interpolate(method='index')

    # Reindex down to only the uniform grid points
    driver_df = driver_df.reindex(uniform_index)

    # Round gear to nearest integer
    driver_df['gear'] = driver_df['gear'].round()
    driver_df['gear'] = driver_df['gear'].astype(int)

    print(driver_df)

def main():

    # Time - Does not need to be normalized
    # Distance - does not need to be normalized

    # Rel_Distance - Is already normalized?

    # Track coordinates - does not need to be normalized

    # RPM - Needs to be normalized
    # Gear - Categorical normalization research
    # Throttle - Needs to be normalized
    # Brake - good
    # DRS: all nan?????
    # Speed Needs to be normalized

    # Acc_x, y, z - needs to be normalized (need to maintain negativity)? how do we maintain negativity with gloabl?? or is it all going to fall roughly in the same

    # query_and_df_creation()
    pickle_df_repop()
    get_max_values()
    # interpolate()
        # uniform_index = np.linspace(0, 1, datapoints_per_lap)
        # print("BEFORE")
        # print(driver_df)
        # driver_df = driver_df.reindex(driver_df.index.union(uniform_index)).interpolate(method='index').reindex(uniform_index)
        # print("AFTER")
        # print(driver_df)
        

if __name__=="__main__":
    main()
