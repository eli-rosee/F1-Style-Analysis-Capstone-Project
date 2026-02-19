import psycopg2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# turn 10 sql query
query = """
SELECT
    r.driver_id,
    r.lap,
    MIN(t.speed) AS min_speed,
    AVG(t.speed) AS avg_speed,
    AVG(t.throttle) AS avg_throttle,
    AVG(CASE WHEN t.brake THEN 1 ELSE 0 END) AS brake_pct
FROM race_lap_data r
JOIN telemetry_data t ON r.tel_index = t.tel_index
WHERE r.race_name = 'CAN'
  AND r.year = 2025
  AND t.distance BETWEEN 2500 AND 2900
GROUP BY r.driver_id, r.lap
ORDER BY r.driver_id, r.lap;
"""

# connect and load to db
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="Team6"
)

df = pd.read_sql(query, conn)
conn.close()

# k means
features = df[["min_speed", "avg_speed", "avg_throttle", "brake_pct"]]

X_scaled = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=17, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# output
print("\nCluster counts:")
print(df["cluster"].value_counts().sort_index())

centers = StandardScaler().fit(features).inverse_transform(
    kmeans.cluster_centers_
)
print("\nCluster centers:")
print(pd.DataFrame(
    centers,
    columns=features.columns
))

df.to_csv("can2025_turn10_kmeans.csv", index=False)
print("\nSaved can2025_turn10_kmeans.csv")
