import psycopg2
from psycopg2 import Error
import json
import pandas as pd    

def connect_to_db():
    # Connect to your postgres DB
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="Team6"
    )
    return conn

def main():
    try:
        conn = connect_to_db()
        print("Connected to the database successfully!")
        cursor = conn.cursor()
        
        # reading the JSON data using json.load()
        file = 'telemetry/Australian_Grand_Prix/ALB/1_tel.json'
        with open(file) as train_file:
            dict_train = json.load(train_file)['tel']

        # converting json dataset from dictionary to dataframe
        train = pd.DataFrame(data=dict_train, index=None, columns=None, dtype=None, copy=None)

        print(train.head(10))

    except (Exception, Error) as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if(conn):
            conn.close()
            cursor.close()
            print("PostgreSQL connection is closed")

if __name__ == "__main__":
    main()
