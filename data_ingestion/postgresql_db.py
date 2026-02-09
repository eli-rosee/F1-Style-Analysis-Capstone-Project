import psycopg2
from psycopg2 import Error

class telemetry_database():
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

    def create_telemetry_data_table(self):
        pass