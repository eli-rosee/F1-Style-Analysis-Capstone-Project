"""
SIMPLIFIED F1 Clustering Script - MULTI-METRIC VERSION
Focuses on ONE turn, tests MULTIPLE metrics
Tests: throttle, brake, speed, rpm, gear
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Configuration
TELEMETRY_DIR = "./telemetry"
RACE_NAME = "Canadian_Grand_Prix"
METRICS_TO_TEST = ['throttle', 'brake', 'speed', 'rpm', 'gear']  # Test all 5
TURN_START = 2800  # Turn 8-9 chicane
TURN_END = 3100
N_CLUSTERS = 2


def load_turn_data(telemetry_dir, race_name, turn_start, turn_end):
    """Load telemetry data for a specific turn only."""
    
    print(f"\nLoading {race_name} - Turn ({turn_start}-{turn_end}m)...")
    
    all_data = []
    race_path = os.path.join(telemetry_dir, race_name)
    
    for driver_code in os.listdir(race_path):
        driver_path = os.path.join(race_path, driver_code)
        
        if not os.path.isdir(driver_path):
            continue
        
        for filename in os.listdir(driver_path):
            if not filename.endswith('_tel.json'):
                continue
            
            lap_number = int(filename.replace('_tel.json', ''))
            filepath = os.path.join(driver_path, filename)
            
            try:
                with open(filepath, 'r') as f:
                    lap_data = json.load(f)
                
                df_lap = pd.DataFrame(lap_data['tel'])
                
                # Filter to turn only
                df_lap = df_lap[(df_lap['distance'] >= turn_start) & 
                               (df_lap['distance'] <= turn_end)]
                
                if len(df_lap) < 5:  # Skip if not enough data
                    continue
                
                df_lap['driver_id'] = driver_code
                df_lap['lap_number'] = lap_number
                df_lap['lap_id'] = f"{driver_code}_lap{lap_number}"
                
                all_data.append(df_lap)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(df):,} data points from {df['lap_id'].nunique()} laps")
    return df


def cluster_turn_multivariate(df, metrics, n_clusters):
    """Cluster laps using MULTIPLE metrics together."""

    print(f"\n{'='*60}")
    print(f"MULTIVARIATE CLUSTERING: {', '.join([m.upper() for m in metrics])}")
    print(f"{'='*60}")

    laps = []
    lap_ids = []
    drivers = []

    fixed_length = 20  # 20 points per metric

    for lap_id in df['lap_id'].unique():
        lap_data = df[df['lap_id'] == lap_id].sort_values('distance')

        if len(lap_data) < 5:
            continue

        lap_features = []

        for metric in metrics:
            if metric not in lap_data.columns:
                print(f"Metric '{metric}' missing, skipping lap.")
                continue

            values = lap_data[metric].values

            resampled = np.interp(
                np.linspace(0, len(values)-1, fixed_length),
                np.arange(len(values)),
                values
            )

            # Scale EACH metric independently
            resampled = StandardScaler().fit_transform(resampled.reshape(-1,1)).flatten()

            lap_features.append(resampled)

        if len(lap_features) != len(metrics):
            continue

        combined_vector = np.concatenate(lap_features)
        laps.append(combined_vector)

        lap_ids.append(lap_id)
        drivers.append(lap_data['driver_id'].iloc[0])

    X = np.array(laps)
    print(f"Feature matrix: {X.shape[0]} laps × {X.shape[1]} features")

    from sklearn.decomposition import PCA

    # Reduce dimensionality
    pca = PCA(n_components=0.95)  # keep 95% variance
    X_reduced = pca.fit_transform(X)

    print(f"PCA reduced dimensions: {X.shape[1]} → {X_reduced.shape[1]}")


    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    print(f"\nClustering Quality:")
    print(f"  Silhouette Score: {sil:.3f}")
    print(f"  Davies-Bouldin Index: {db:.3f}")

    results = pd.DataFrame({
        'lap_id': lap_ids,
        'driver_id': drivers,
        'cluster': labels
    })

    driver_profiles = (
        results.groupby('driver_id')['cluster']
        .value_counts(normalize=True)
        .unstack(fill_value=0) * 100
    )

    return results, driver_profiles, X, labels, sil, db



def visualize_results(X, labels, driver_profiles, metric):
    """Create simple visualizations."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    axes[0].bar(unique, counts, color=colors[:len(unique)])
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Number of Laps', fontsize=12)
    axes[0].set_title(f'{metric.upper()} Clusters - Turn 8-9', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Top drivers by style diversity
    driver_entropy = -(driver_profiles / 100 * np.log(driver_profiles / 100 + 1e-10)).sum(axis=1)
    top_diverse = driver_entropy.nlargest(10)
    
    axes[1].barh(range(len(top_diverse)), top_diverse.values, color='#1f77b4')
    axes[1].set_yticks(range(len(top_diverse)))
    axes[1].set_yticklabels(top_diverse.index)
    axes[1].set_xlabel('Style Diversity (Entropy)', fontsize=12)
    axes[1].set_title('Most Adaptive Drivers', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'turn_clustering_{metric}.png', dpi=300, bbox_inches='tight')
    print(f"  Visualization saved: turn_clustering_{metric}.png")


def create_comparison_summary(all_results):
    """Create a comparison table and visualization across all metrics."""
    
    print("\n" + "="*60)
    print("MULTI-METRIC COMPARISON SUMMARY")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for metric, data in all_results.items():
        if data['silhouette'] is not None:
            comparison_data.append({
                'Metric': metric.upper(),
                'Silhouette': data['silhouette'],
                'Davies-Bouldin': data['davies_bouldin'],
                'Dominant_Cluster_%': data['dominant_pct']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Silhouette', ascending=False)
    
    print("\nRanked by Clustering Quality (Silhouette Score):")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('metric_comparison_turn89.csv', index=False)
    print("\nComparison saved: metric_comparison_turn89.csv")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Silhouette scores
    metrics = comparison_df['Metric'].values
    sil_scores = comparison_df['Silhouette'].values
    colors_sil = ['#2ecc71' if s > 0.7 else '#f39c12' if s > 0.5 else '#e74c3c' for s in sil_scores]
    
    axes[0].barh(range(len(metrics)), sil_scores, color=colors_sil)
    axes[0].set_yticks(range(len(metrics)))
    axes[0].set_yticklabels(metrics)
    axes[0].set_xlabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Clustering Quality by Metric', fontsize=14, fontweight='bold')
    axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Good (>0.5)')
    axes[0].axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.7)')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Davies-Bouldin Index (lower is better)
    db_scores = comparison_df['Davies-Bouldin'].values
    colors_db = ['#2ecc71' if d < 1.0 else '#f39c12' if d < 2.0 else '#e74c3c' for d in db_scores]
    
    axes[1].barh(range(len(metrics)), db_scores, color=colors_db)
    axes[1].set_yticks(range(len(metrics)))
    axes[1].set_yticklabels(metrics)
    axes[1].set_xlabel('Davies-Bouldin Index', fontsize=12)
    axes[1].set_title('Cluster Separation (Lower = Better)', fontsize=14, fontweight='bold')
    axes[1].axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Good (<1.0)')
    axes[1].axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, label='Moderate (<2.0)')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metric_comparison_visualization.png', dpi=300, bbox_inches='tight')
    print("Comparison visualization saved: metric_comparison_visualization.png")
    
    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    best_metric = comparison_df.iloc[0]['Metric']
    best_sil = comparison_df.iloc[0]['Silhouette']
    print(f"\nBest Metric: {best_metric} (Silhouette: {best_sil:.3f})")
    
    excellent = comparison_df[comparison_df['Silhouette'] > 0.7]
    if len(excellent) > 0:
        print(f"\n Excellent metrics (Silhouette > 0.7):")
        for _, row in excellent.iterrows():
            print(f"   - {row['Metric']}: {row['Silhouette']:.3f}")
    
    good = comparison_df[(comparison_df['Silhouette'] > 0.5) & (comparison_df['Silhouette'] <= 0.7)]
    if len(good) > 0:
        print(f"\n Good metrics (Silhouette 0.5-0.7):")
        for _, row in good.iterrows():
            print(f"   - {row['Metric']}: {row['Silhouette']:.3f}")
    
    poor = comparison_df[comparison_df['Silhouette'] <= 0.5]
    if len(poor) > 0:
        print(f"\n Poor metrics (Silhouette < 0.5):")
        for _, row in poor.iterrows():
            print(f"   - {row['Metric']}: {row['Silhouette']:.3f}")


def main():
    print("="*60)
    print("F1 Single-Turn Multi-Metric Clustering")
    print("="*60)
    print(f"Turn: {TURN_START}-{TURN_END}m (Turn 8-9 Chicane)")
    print(f"Metrics to test: {', '.join([m.upper() for m in METRICS_TO_TEST])}")
    print(f"Clusters: {N_CLUSTERS}")
    print("="*60)
    
    # Load data once
    df = load_turn_data(TELEMETRY_DIR, RACE_NAME, TURN_START, TURN_END)
    
    # Store results for all metrics
    all_results = {}
    
    # Cluster on each metric
    try:
        results, driver_profiles, X, labels, sil, db = cluster_turn_multivariate(
            df,
            METRICS_TO_TEST,
            N_CLUSTERS
        )

        if results is not None:

            metric_name = "multivariate"

            # Visualize
            visualize_results(X, labels, driver_profiles, metric_name)

            # Save driver profiles
            output_file = f'driver_profiles_{metric_name}_turn89.csv'
            driver_profiles.to_csv(output_file)
            print(f"  Driver profiles saved: {output_file}")

            # Calculate dominant cluster percentage
            unique, counts = np.unique(labels, return_counts=True)
            dominant_pct = (counts.max() / len(labels)) * 100

            # Store results
            all_results[metric_name] = {
                'silhouette': sil,
                'davies_bouldin': db,
                'dominant_pct': dominant_pct,
                'driver_profiles': driver_profiles
            }

            print(" MULTIVARIATE clustering complete!\n")

    except Exception as e:
        print(f" Error in multivariate clustering: {e}\n")
    
    # Create comparison summary
    if all_results:
        create_comparison_summary(all_results)
        
        # Print top 5 drivers for best metric
        best_metric = max(all_results.items(), key=lambda x: x[1]['silhouette'])[0]
        print("\n" + "="*60)
        print(f"Top 10 Driver Profiles - {best_metric.upper()} (Best Metric)")
        print("="*60)
        print(all_results[best_metric]['driver_profiles'].head(10).round(1))
    else:
        print("\n No metrics successfully clustered!")


if __name__ == "__main__":
    main()
