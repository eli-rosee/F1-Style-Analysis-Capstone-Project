import psycopg2
from psycopg2 import Error
import json
import pandas as pd
import os   

class telemetry_database():
    host="localhost"
    database="postgres"
    user="postgres"
    password="Team6"

    schema_mapping = {"x":"track_coordinate_x", "y":"track_coordinate_y", "z":"track_coordinate_z"}

    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                host="localhost",
                database="postgres",
                user="postgres",
                password="Team6"
            )

            self.cursor = self.conn.cursor()
            self.conn.autocommit = True

            print("Connected to the database successfully!")

        except (Exception, Error) as e:
            print(f"Error connecting to the database: {e}")

    def __del__(self):
        try:
            self.conn.close()
            self.cursor.close()

            print("Successfully closed connection to the database.")

        except (Exception, Error) as e:
            print(f"Error closing the database: {e}")

    def process_tel_file(self, filepath):
        with open(filepath) as train_file:
            dict_train = json.load(train_file)['tel']

        train = pd.DataFrame(data=dict_train, index=None, columns=None, dtype=None, copy=None)
        train.rename(columns=telemetry_database.schema_mapping, inplace=True)
        print(train.columns)
        train.drop(['dataKey'], axis=1, inplace=True)

        print(train.head(10))

        return train

def main():
    db = telemetry_database()
    file = 'telemetry/Australian_Grand_Prix/ALB/1_tel.json'

    tel_df = db.process_tel_file(file)

    # tel_df.to_sql('telemetry_data', con=db.conn, if_exists="append", index=False)

if __name__ == "__main__":
    main()
