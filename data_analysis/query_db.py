import psycopg2
import pandas as pd
import json

CACHE_FILE = 'cache/race_metadata_cache.json'

with open(CACHE_FILE) as f:
    _cache = json.load(f)

# Maps race name -> table name strings from the cache
TEL_TABLE = {}
META_TABLE = {}
for race, data in _cache['races'].items():
    TEL_TABLE[race] = data['telemetry_db_code']
    META_TABLE[race] = data['metadata_db_code']


class TelemetryDatabase:

    def __init__(self):
        self.conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="Team6")
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    # Fetches all telemetry rows for a given driver and lap from the correct DB tables
    def fetch_driver_telemetry_by_lap(self, race_name, driver, columns, lap_num):
        tel_table = TEL_TABLE[race_name]
        meta_table = META_TABLE[race_name]
        col_str = ', '.join(columns)

        self.cursor.execute(
            f"SELECT {col_str} FROM {tel_table} WHERE tel_index IN (SELECT tel_index FROM {meta_table} WHERE driver_id = %s AND lap = %s) ORDER BY rel_distance",
            (driver, lap_num)
        )
        rows = self.cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)
