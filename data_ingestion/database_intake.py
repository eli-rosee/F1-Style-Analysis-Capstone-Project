import psycopg2
from psycopg2 import Error
import json
import pandas as pd
import os   
import glob
import postgresql_db
from sqlalchemy import create_engine
from postgresql_db import telemetry_database

schema_mapping = {"x":"track_coordinate_x", "y":"track_coordinate_y", "z":"track_coordinate_z"}

#reads a filepath for a json file and returns it as a pandas dataframe
def process_tel_file(filepath):
    with open(filepath) as train_file:
        dict_train = json.load(train_file)['tel']

    train = pd.DataFrame(data=dict_train, index=None, columns=None, dtype=None, copy=None)
    train.rename(columns=schema_mapping, inplace=True)
    train.drop(['dataKey'], axis=1, inplace=True)

    return train

#given a race name (string), all telemetry files will be returned as a list of pandas dataframes
def convert_race_to_dataframe_list(race_name):

    base_path = os.path.join('telemetry', race_name)
    dataframe_list = []

    drivers = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for driver in drivers:
        search_pattern = os.path.join(base_path, driver, "*_tel.json")
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

def main():
    #connect to database
    #conn_string = f'postgresql://{telemetry_database.host}:{telemetry_database.password}@{telemetry_database.host}/telemetry_data'
    #db = create_engine(conn_string)
    #conn = db.connect()
    
    #convert .json to pandas dataframe list
    dataframe_list = convert_race_to_dataframe_list("Australian_Grand_Prix")

    #print("Saving pandas dataframes to race_data.csv")
    #df.to_csv("race_data.csv")

    '''
    #attempt to add dataframe to the database
    try:
        tel_df.to_sql('telemetry_data', con=conn, if_exists="append", index=False)
    except (Exception, Error) as e:
        print(f"Error: {e}")
    '''


if __name__ == "__main__":
    main()