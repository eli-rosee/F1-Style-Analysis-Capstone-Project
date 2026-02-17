from psycopg2 import Error
import json
import pandas as pd
import os
import glob
import fastf1
from sqlalchemy import create_engine
from postgresql_db import telemetry_database

schema_mapping = {"x":"track_coordinate_x", "y":"track_coordinate_y", "z":"track_coordinate_z"}

#create dictionary that maps a race name to a 3 digit code
race_code_map = {"Belgian_Grand_Prix" : "BEL", "Chinese_Grand_Prix" : "CHN", "Hungarian_Grand_Prix" : "HUN", "Japanese_Grand_Prix" : "JPN", "Dutch_Grand_Prix" : "NED",
                 "Bahrain_Grand_Prix" : "BAH", "Italian_Grand_Prix" : "ITA", "Saudi_Arabian_Grand_Prix" : "SAU", "Azerbaijan_Grand_Prix" : "AZE", "Miami_Grand_Prix" : "MIA",
                 "Singapore_Grand_Prix" : "SIN", "Emilia_Romagna_Grand_Prix" : "EMI", "United_States_Grand_Prix" : "USA", "Monaco_Grand_Prix" : "MON", "Mexico_City_Grand_Prix" : "MEX",
                 "Spanish_Grand_Prix" : "ESP", "São_Paulo_Grand_Prix" : "SAO", "Canadian_Grand_Prix" : "CAN", "Las_Vegas_Grand_Prix" : "LAS", "Australian_Grand_Prix" : "AUS",
                 "Qatar_Grand_Prix" : "QAT", "British_Grand_Prix" : "GBR", "Abu_Dhabi_Grand_Prix" : "ABU",
                 }

#given a race name, return a list of drivers
def get_drivers_from_race(race_name):
    try:
        base_path = os.path.join('telemetry', race_name)
        drivers = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except:
        return [] #return empty list if no data is found for a certain race

    return drivers

#reads a filepath for a json file and returns it as a pandas dataframe
def process_tel_file(filepath, lap_num, driver_name):
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
        lap_num = int(os.path.basename(file).split("_")[0])
        df = process_tel_file(file, lap_num, driver_name)
        df["driver_id"] = driver_name
        df["lap"] = lap_num
        dataframe_list.append(df)

    if dataframe_list:
        #return a list of pandas dataframes
        return dataframe_list

        #if you would rather have this be one massive dataframe, change this return to:
        #pd.concat(all_race_data, ignore_index=True)

    #if no file directory is found, return empty list
    return []

def main():
    #get list of all races on the schedule
    schedule = fastf1.get_event_schedule(2025)['EventName']
    
    year = 2025  # set this to whatever year your data is

    #fill raceNameList and replace spaces with underlines
    raceNameList = []
    for i in schedule:
        s = i.replace(" ", "_")
        raceNameList.append(s)

    # DB connection SQLAlchemy engine
    user = telemetry_database.user
    password = telemetry_database.password
    host = telemetry_database.host
    dbname = telemetry_database.database
    port = 5432

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
    
    #loop through all races for a certain season
    for raceName in raceNameList:

        drivers = get_drivers_from_race(raceName)

        VALID_COLUMNS = [
            "tel_index",
            "time", "distance", "rel_distance",
            "track_coordinate_x", "track_coordinate_y", "track_coordinate_z",
            "rpm", "gear", "throttle", "brake",
            "speed", "acc_x", "acc_y", "acc_z"
        ]

        # Get next ids, lap_data_id is not auto generated
        with engine.connect() as conn:
            next_tel_index = pd.read_sql(
                "SELECT COALESCE(MAX(tel_index), 0) + 1 AS nxt FROM telemetry_data",
                conn
            ).iloc[0]["nxt"]

            next_lap_data_id = pd.read_sql(
                "SELECT COALESCE(MAX(lap_data_id), 0) + 1 AS nxt FROM race_lap_data",
                conn
            ).iloc[0]["nxt"]

        inserted_tel = 0
        inserted_lap = 0

        #try to insert data from all races, if data for a race is not present in the directory, then continue
        try:

            for driver in drivers:
                dataframe_list = convert_race_to_dataframe_list(raceName, driver)

                for i, df in enumerate(dataframe_list, start=1):
                    # Saving these before filtering columns
                    driver_id = df["driver_id"].iloc[0]
                    lap_num = int(df["lap"].iloc[0])

                    # keep only expected telemetry columns (copy prevents SettingWithCopyWarning)
                    df = df[[c for c in df.columns if c in VALID_COLUMNS]].copy()

                    # normalize brake
                    if "brake" in df.columns:
                        df["brake"] = df["brake"].astype(bool)

                    # normalize gear (0-8 per DB check)
                    if "gear" in df.columns:
                        df["gear"] = pd.to_numeric(df["gear"], errors="coerce")
                        df.loc[~df["gear"].between(0, 8), "gear"] = pd.NA

                    # validate throttle (0-100 per DB check)
                    if "throttle" in df.columns:
                        df["throttle"] = pd.to_numeric(df["throttle"], errors="coerce")
                        df.loc[~df["throttle"].between(0, 100), "throttle"] = pd.NA

                    # assign tel_index for every telemetry row in this df
                    n = len(df)
                    df["tel_index"] = range(next_tel_index, next_tel_index + n)

                    # build race_lap_data rows (one per telemetry row)
                    lap_df = pd.DataFrame({
                        "lap_data_id": range(next_lap_data_id, next_lap_data_id + n),
                        "driver_id": [driver_id] * n,
                        "lap": [lap_num] * n,
                        "race_name": [race_code_map[raceName]] * n,
                        "year": [year] * n,
                        "tel_index": df["tel_index"].values
                    })

                    try:
                        # insert telemetry rows
                        df.to_sql("telemetry_data", con=engine, if_exists="append", index=False, method="multi")
                        inserted_tel += n
                        next_tel_index += n

                        # insert race_lap_data rows
                        lap_df.to_sql("race_lap_data", con=engine, if_exists="append", index=False, method="multi")
                        inserted_lap += n
                        next_lap_data_id += n

                        print(f"{driver} lap {lap_num}: +{n} telemetry, +{n} race_lap_data (tot_tel={inserted_tel}, tot_lap={inserted_lap})")

                    except Exception as e:
                        msg = str(e).split("\n")[0]
                        print(f"INSERT FAILED for {driver}: {msg}")
                        return
        
        except Exception as e:
            print(f"Skipping: No data found for {raceName}")
            continue

    print(f"Done. Inserted {inserted_tel} telemetry rows and {inserted_lap} race_lap_data rows.")
    
if __name__ == "__main__":
    main()
