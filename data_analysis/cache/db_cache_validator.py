import json
import psycopg2

CACHE_FILE = 'f1_cache.json'

conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="Team6")
cursor = conn.cursor()

with open(CACHE_FILE) as f:
    cache = json.load(f)

for race_name, race_data in cache['races'].items():
    table = race_data['metadata_db_code']
    expected = race_data['driver_laps']

    print(f"\n{race_name}")

    for driver, expected_laps in expected.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE driver_id = %s", (driver,))
        actual = cursor.fetchone()[0]

        if actual == 0:
            print(f"  MISSING  {driver}: expected {expected_laps} laps, got nothing")
        elif actual == expected_laps or actual == expected_laps + 1:
            print(f"  OK       {driver}: {actual} laps")
        elif actual > expected_laps + 1:
            print(f"  EXTRA    {driver}: expected {expected_laps}, got {actual}")
        else:
            print(f"  SHORT    {driver}: expected {expected_laps}, got {actual}")

    cursor.execute(f"SELECT driver_id, COUNT(*) FROM {table} GROUP BY driver_id")
    for driver, actual in cursor.fetchall():
        if driver not in expected:
            print(f"  EXTRA    {driver}: in DB ({actual} laps) but not in cache")

cursor.close()
conn.close()
