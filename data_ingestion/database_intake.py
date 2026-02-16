from psycopg2 import Error
import json
import pandas as pd
import os
import glob
from sqlalchemy import create_engine
from postgresql_db import telemetry_database

schema_mapping = {"x":"track_coordinate_x", "y":"track_coordinate_y", "z":"track_coordinate_z"}

def get_drivers_from_race(race_name):
    base_path = os.path.join('telemetry', race_name)
    drivers = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    return drivers

#reads a filepath for a json file and returns it as a pandas dataframe
def process_tel_file(filepath):
    with open(filepath) as train_file:
        dict_train = json.load(train_file)['tel']

    train = pd.DataFrame(data=dict_train, index=None, columns=None, dtype=None, copy=None)
    train.rename(columns=schema_mapping, inplace=True)
    train.drop(['dataKey'], axis=1, inplace=True)

    return train

#given a race name (string), all telemetry files will be returned as a list of pandas dataframes
def convert_race_to_dataframe_list(race_name, driver_name):

    base_path = os.path.join('telemetry', race_name)

    #declare empty list to store pandas dataframes
    dataframe_list = []

    search_pattern = os.path.join(base_path, driver_name, "*_tel.json")
    file_list = glob.glob(search_pattern)
    
    #Loop through the files and place the DataFrames into a list
    for file in file_list:
        print(f"Reading: {file}")
        df = process_tel_file(file)
        dataframe_list.append(df)

    if dataframe_list:
        #return a list of pandas dataframes
        return dataframe_list 
        
        #if you would rather have this be one massive dataframe, change this return to: 
        #pd.concat(all_race_data, ignore_index=True)
    
    #if no file directory is found, return empty list
    return []

# def insert_race_database(race_name, year, )

def main():
    raceName = "Australian_Grand_Prix"

    # DB connection SQLAlchemy engine
    user = telemetry_database.user
    password = telemetry_database.password
    host = telemetry_database.host
    dbname = telemetry_database.database
    port = 5432

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")

    drivers = get_drivers_from_race(raceName)

    for driver in drivers:
        dataframe_list = convert_race_to_dataframe_list(raceName, driver)

    VALID_COLUMNS = [
        "time", "distance", "rel_distance",
        "track_coordinate_x", "track_coordinate_y", "track_coordinate_z",
        "rpm", "gear", "throttle", "brake",
        "speed", "acc_x", "acc_y", "acc_z"
    ]

    inserted = 0
    for i, df in enumerate(dataframe_list, start=1):
        # keep only expected columns (made real copy to stop SettingWithCopyWarning)
        df = df[[c for c in df.columns if c in VALID_COLUMNS]].copy()

        # ensure boolean for brake
        if "brake" in df.columns:
            df["brake"] = df["brake"].astype(bool)

        # normalize gear; numeric, replace invalid with NULL, enforce 0-8
        if "gear" in df.columns:
            df["gear"] = pd.to_numeric(df["gear"], errors="coerce")
            df.loc[~df["gear"].between(0, 8), "gear"] = pd.NA

        # validate throttle percentages; coerce invalids to null
        if "throttle" in df.columns:
            df["throttle"] = pd.to_numeric(df["throttle"], errors="coerce")
            df.loc[~df["throttle"].between(0, 100), "throttle"] = pd.NA

        try:
            df.to_sql(
                "telemetry_data",
                con=engine,
                if_exists="append",
                index=False,
                method="multi"
            )
            inserted += len(df)
            print(f"[{i}/{len(dataframe_list)}] Inserted {len(df)} rows (total={inserted})")
        except Exception as e:
            # stop the giant SQL spam
            msg = str(e).split("\n")[0]
            print(f"INSERT FAILED on chunk {i}: {msg}")
            break

    print(f"Done. Inserted {inserted} telemetry rows.")

if __name__ == "__main__":
    main()
