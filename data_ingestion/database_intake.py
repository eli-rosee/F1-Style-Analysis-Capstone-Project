import psycopg2
from psycopg2 import Error
import json
import pandas as pd
import os   
import postgresql_db

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
    db = postgresql_db.telemetry_database()
    file = 'telemetry/Australian_Grand_Prix/ALB/1_tel.json'

    tel_df = process_tel_file(file)

    # tel_df.to_sql('telemetry_data', con=db.conn, if_exists="append", index=False)

if __name__ == "__main__":
    main()
