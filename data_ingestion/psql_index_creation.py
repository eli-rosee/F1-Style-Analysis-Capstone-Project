import psycopg2
import json

CACHE_FILE = 'cache/race_metadata_cache.json'

with open(CACHE_FILE) as f:
    cache = json.load(f)

conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="Team6")
cursor = conn.cursor()

for race, data in cache['races'].items():
    meta = data['metadata_db_code']
    tel = data['telemetry_db_code']
    print(f"Indexing {race}...")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS ON {meta} (driver_id, lap)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS ON {tel} (tel_index)")

conn.commit()
cursor.close()
conn.close()
print("Done.")
