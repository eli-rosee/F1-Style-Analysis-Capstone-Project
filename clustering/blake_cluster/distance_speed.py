import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="Team6"
)

query = """
SELECT
    t.distance,
    t.speed
FROM race_lap_data r
JOIN telemetry_data t ON r.tel_index = t.tel_index
WHERE r.race_name = 'CAN'
  AND r.year = 2025
  AND r.driver_id = 'VER'
  AND r.lap = 10
ORDER BY t.distance;
"""

df = pd.read_sql(query, conn)
conn.close()

print(df.head())

# plotting
plt.figure(figsize=(10,4))
plt.plot(df["distance"], df["speed"])
plt.xlabel("Distance")
plt.ylabel("Speed")
plt.title("Speed vs Distance – CAN 2025 – VERLap 10")

plt.tight_layout()
plt.savefig("speed_vs_distance.png")
plt.close()

print("Saved plot to speed_vs_distance.png")
