import psycopg2
from psycopg2 import Error
import json
import pandas as pd
import os   
import postgresql_db
from sqlalchemy import create_engine
from postgresql_db import telemetry_database

schema_mapping = {"x":"track_coordinate_x", "y":"track_coordinate_y", "z":"track_coordinate_z"}

def process_tel_file(filepath):
    with open(filepath) as train_file:
        dict_train = json.load(train_file)['tel']

    train = pd.DataFrame(data=dict_train, index=None, columns=None, dtype=None, copy=None)
    train.rename(columns=schema_mapping, inplace=True)
    print(train.columns)
    train.drop(['dataKey'], axis=1, inplace=True)

    print(train.head(10))

    return train

def main():
    conn_string = f'postgres://{telemetry_database.host}:{telemetry_database.password}@{telemetry_database.host}/telemetry_data'
    db = create_engine(conn_string)
    conn = db.connect()

    file = 'telemetry/Australian_Grand_Prix/ALB/1_tel.json'

    tel_df = process_tel_file(file)

    try:
        tel_df.to_sql('telemetry_data', con=conn, if_exists="append", index=False)
    except (Exception, Error) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
