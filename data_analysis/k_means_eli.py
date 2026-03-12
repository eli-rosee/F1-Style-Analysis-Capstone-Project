import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from race_data import RaceData

def cluster(interpolated_df_dict, drivers, norm_tel_columns):
    print("Clustering data...")
    all_laps = []
    for driver in drivers:
        for lap_df in interpolated_df_dict[driver]:
            all_laps.append(lap_df[norm_tel_columns].values.flatten())

    X = np.array(all_laps)

    km = KMeans(n_clusters=4, random_state=42)
    labels = km.fit_predict(X)

    return labels


def attach_labels(interpolated_df_dict, drivers, labels):
    print("Attaching cluster labels to dataframes...")
    lap_index = 0
    for driver in drivers:
        for lap_df in interpolated_df_dict[driver]:
            lap_df['cluster_label'] = labels[lap_index]
            lap_index += 1


def elbow_plot(interpolated_df_dict, drivers, norm_tel_columns):
    print("Generating elbow plot...")
    all_laps = []
    for driver in drivers:
        for lap_df in interpolated_df_dict[driver]:
            all_laps.append(lap_df[norm_tel_columns].values.flatten())

    X = np.array(all_laps)

    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.savefig('elbow_plot.png')


def visualize_clusters(interpolated_df_dict, drivers, norm_tel_columns):
    cluster_data = {}
    for driver in drivers:
        for lap_df in interpolated_df_dict[driver]:
            label = lap_df['cluster_label'].iloc[0]
            if label not in cluster_data:
                cluster_data[label] = []
            cluster_data[label].append(lap_df[norm_tel_columns].values)

    fig, axes = plt.subplots(len(norm_tel_columns), 1, figsize=(12, 20))
    for i, feature in enumerate(norm_tel_columns):
        for label, laps in cluster_data.items():
            mean_curve = np.mean([lap[:, i] for lap in laps], axis=0)
            axes[i].plot(mean_curve, label=f'Cluster {label}')
        axes[i].set_title(feature)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('cluster_visualization.png')


def cluster_statistics(interpolated_df_dict, drivers, norm_tel_columns):
    print("\n========== CLUSTER STATISTICS ==========")

    lap_records = []
    for driver in drivers:
        for lap_num, lap_df in enumerate(interpolated_df_dict[driver]):
            label = lap_df['cluster_label'].iloc[0]
            lap_records.append({'driver': driver, 'lap': lap_num + 1, 'cluster': label})

    records_df = pd.DataFrame(lap_records)

    for cluster_id in sorted(records_df['cluster'].unique()):
        cluster_laps = [
            interpolated_df_dict[row['driver']][row['lap'] - 1]
            for _, row in records_df[records_df['cluster'] == cluster_id].iterrows()
        ]

        count = len(cluster_laps)
        total = len(records_df)
        all_data = pd.concat(cluster_laps)[norm_tel_columns]

        print(f"\n--- Cluster {cluster_id} ---")
        print(f"  Lap count : {count} ({100 * count / total:.1f}% of all laps)")
        print(f"  Mean telemetry:")
        for col in norm_tel_columns:
            print(f"    {col:<22} mean={all_data[col].mean():.3f}  std={all_data[col].std():.3f}")

    print("\n========== DRIVER CLUSTER BREAKDOWN ==========")
    cluster_ids = sorted(records_df['cluster'].unique())
    header = f"{'DRIVER':<8}" + "".join(f"  C{c}%" for c in cluster_ids)
    print(header)

    for driver in drivers:
        driver_laps = records_df[records_df['driver'] == driver]
        total = len(driver_laps)
        row = f"{driver:<8} ({total} laps)"
        for c in cluster_ids:
            pct = 100 * len(driver_laps[driver_laps['cluster'] == c]) / total
            row += f"  {pct:>4.1f}"
        print(row)

def main():
    race_name = 'Canadian_Grand_Prix'
    race = RaceData(race_name)

    drivers = race.drivers
    interp = race.interp_dict

    labels = cluster(interp, drivers, race.norm_columns)
    attach_labels(interp, drivers, labels)
    # elbow_plot(interp, drivers, race.norm_columns)
    visualize_clusters(interp, drivers, race.norm_columns)
    cluster_statistics(interp, drivers, race.norm_columns)

if __name__ == '__main__':
    main()
