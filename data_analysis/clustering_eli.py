import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from race_data import RaceData
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture

# ── Toggle this to switch between raw telemetry and PCA-reduced data ──
USE_PCA = True


def _build_matrix(data_dict, drivers):
    all_laps = []
    for driver in drivers:
        for lap in data_dict[driver]:
            if isinstance(lap, np.ndarray):
                all_laps.append(lap.flatten())
            else:
                all_laps.append(lap.values.flatten())
    return np.array(all_laps)

def k_means_cluster(data_dict, drivers, cluster_num):
    print("\nClustering data (KMeans)...\n")
    X = _build_matrix(data_dict, drivers)

    km = KMeans(n_clusters=cluster_num, random_state=42)
    labels = km.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)
    cah_score = calinski_harabasz_score(X, labels)

    return labels, sil_score, dbi_score, cah_score

def gmm_cluster(data_dict, drivers, cluster_num):
    print("\nClustering data (GMM)...\n")
    X = _build_matrix(data_dict, drivers)

    gmm = GaussianMixture(n_components=cluster_num, random_state=42)
    labels = gmm.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)
    cah_score = calinski_harabasz_score(X, labels)

    return labels, sil_score, dbi_score, cah_score

def attach_labels(interp_dict, drivers, labels):
    lap_index = 0
    for driver in drivers:
        for lap_df in interp_dict[driver]:
            lap_df['cluster_label'] = labels[lap_index]
            lap_index += 1

def driver_cluster_distribution(interp_dict, drivers):
    for driver in drivers:
        cluster_counts = {}

        for lap_df in interp_dict[driver]:
            label = lap_df['cluster_label'].iloc[0]
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        total = sum(cluster_counts.values())
        print(f"{driver} Cluster distribution")

        for cluster_id in sorted(cluster_counts.keys()):
            proportion = cluster_counts[cluster_id] / total
            print(f"  cluster {cluster_id}: {proportion:.2f}")
    
    print()

def cluster_mean_telemetry(interp_dict, drivers, norm_tel_columns):
    cluster_laps = {}
    for driver in drivers:
        for lap_df in interp_dict[driver]:
            label = lap_df['cluster_label'].iloc[0]
            if label not in cluster_laps:
                cluster_laps[label] = []
            cluster_laps[label].append(lap_df)

    for cluster_id in sorted(cluster_laps.keys()):
        laps = cluster_laps[cluster_id]
        all_data = pd.concat(laps)
        print(f"Cluster {cluster_id} ({len(laps)} laps)")
        for col in norm_tel_columns:
            mean = all_data[col].mean()
            print(f"  {col}: {mean:.3f}")


def visualize_clusters(interp_dict, reduced_dict, drivers):
    pc1_vals = []
    pc2_vals = []
    labels = []

    for driver in drivers:
        for lap_df, lap_arr in zip(interp_dict[driver], reduced_dict[driver]):
            pc1_vals.append(lap_arr[:, 0].mean())
            pc2_vals.append(lap_arr[:, 1].mean())
            labels.append(lap_df['cluster_label'].iloc[0])

    plt.figure()
    plt.scatter(pc1_vals, pc2_vals, c=labels, cmap='tab10')
    plt.colorbar(label='Cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('clusters.png')
    plt.show()

def test_cluster_sizes(interp, drivers):
    k_vals = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for k in k_vals:
        print(f"\nTESTING K = {k} clusters...\n")
        lables, sil_score, dbi_score, cah_score = k_means_cluster(interp, drivers, cluster_num=k)
        print(f"Silhouette Score: {sil_score}")
        print(f"Davies Bouldin Index Score: {dbi_score}")
        print(f"Calinski Harabasz Score: {cah_score}")


def main():
    race_name = 'Canadian_Grand_Prix'
    race = RaceData(race_name)

    drivers = race.drivers
    data_dict = {}

    if USE_PCA:
        data_dict = race.reduced_dict
    else:
        data_dict = race.interp_dict

    # test_cluster_sizes(data_dict, drivers)

    labels, sil, dbi, cah = k_means_cluster(data_dict, drivers, cluster_num=5)

    attach_labels(race.interp_dict, drivers, labels)

    driver_cluster_distribution(race.interp_dict, drivers)
    cluster_mean_telemetry(race.interp_dict, drivers, race.norm_columns)
    visualize_clusters(race.interp_dict, race.reduced_dict, drivers)

    print(f"\nSilhouette: {sil:.4f}")
    print(f"Davies Bouldin: {dbi:.4f}")
    print(f"Calinski Harabasz: {cah:.4f}")

if __name__ == '__main__':
    main()
