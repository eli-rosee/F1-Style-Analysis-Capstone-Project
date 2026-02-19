"""
K-Means Clustering Pipeline for F1 Telemetry Data
Formula 1 Telemetry Insights Visual Dashboard
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import json
import warnings
warnings.filterwarnings('ignore')


class F1TelemetryKMeans:
    """
    K-Means clustering implementation for F1 driver throttle behavior analysis.
    """
    
    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize the K-Means clustering pipeline.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_centers = None
        self.metrics = {}
        
    def load_telemetry_data(self, filepath_or_dataframe):
        """
        Load telemetry data from JSON file or DataFrame.
        
        Parameters:
        -----------
        filepath_or_dataframe : str or DataFrame
            Path to JSON file or pandas DataFrame
            
        Returns:
        --------
        DataFrame with telemetry data
        """
        if isinstance(filepath_or_dataframe, str):
            with open(filepath_or_dataframe, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            df = filepath_or_dataframe.copy()
            
        print(f"Loaded telemetry data: {df.shape[0]} records")
        return df
    
    def preprocess_data(self, df, telemetry_column='throttle', 
                       distance_column='distance', lap_identifier='lap_id'):
        """
        Preprocess telemetry data for clustering.
        - Resamples to fixed distance intervals
        - Normalizes using z-scores
        - Reshapes for clustering
        
        Parameters:
        -----------
        df : DataFrame
            Raw telemetry data
        telemetry_column : str
            Column containing the metric to cluster (e.g., 'throttle')
        distance_column : str
            Column containing distance information
        lap_identifier : str
            Column identifying unique laps
            
        Returns:
        --------
        Tuple of (feature_matrix, lap_ids, metadata)
        """
        print("\n=== Preprocessing Data ===")
        
        # Group by lap
        unique_laps = df[lap_identifier].unique()
        print(f"Number of unique laps: {len(unique_laps)}")
        
        # Resample each lap to fixed distance intervals
        resampled_laps = []
        lap_ids = []
        metadata = []
        
        # Define fixed distance intervals (e.g., every 10 meters)
        distance_interval = 10  # meters
        
        for lap_id in unique_laps:
            lap_data = df[df[lap_identifier] == lap_id].copy()
            
            # Skip laps with insufficient data
            if len(lap_data) < 10:
                continue
                
            # Sort by distance
            lap_data = lap_data.sort_values(distance_column)
            
            # Create distance bins
            min_dist = lap_data[distance_column].min()
            max_dist = lap_data[distance_column].max()
            distance_bins = np.arange(min_dist, max_dist, distance_interval)
            
            # Resample throttle data to fixed intervals
            resampled_throttle = np.interp(
                distance_bins,
                lap_data[distance_column].values,
                lap_data[telemetry_column].values
            )
            
            resampled_laps.append(resampled_throttle)
            lap_ids.append(lap_id)
            
            # Store metadata (optional: driver, speed, etc.)
            metadata.append({
                'lap_id': lap_id,
                'mean_throttle': resampled_throttle.mean(),
                'std_throttle': resampled_throttle.std(),
                'max_throttle': resampled_throttle.max()
            })
        
        print(f"Laps after filtering: {len(resampled_laps)}")
        
        # Ensure all laps have same length by truncating or padding
        min_length = min(len(lap) for lap in resampled_laps)
        resampled_laps = [lap[:min_length] for lap in resampled_laps]
        
        # Convert to numpy array (samples x features)
        feature_matrix = np.array(resampled_laps)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Z-score normalization per lap
        print("Applying z-score normalization...")
        feature_matrix_normalized = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix_normalized, lap_ids, pd.DataFrame(metadata)
    
    def detect_outliers(self, feature_matrix, threshold=3):
        """
        Detect and optionally remove outlier laps using z-score method.
        
        Parameters:
        -----------
        feature_matrix : ndarray
            Normalized feature matrix
        threshold : float
            Z-score threshold for outlier detection
            
        Returns:
        --------
        Tuple of (cleaned_matrix, outlier_indices)
        """
        print("\n=== Outlier Detection ===")
        
        # Calculate mean distance from centroid for each lap
        lap_means = feature_matrix.mean(axis=1)
        lap_std = feature_matrix.std(axis=1)
        
        # Z-scores for lap means
        z_scores = np.abs((lap_means - lap_means.mean()) / lap_means.std())
        
        outliers = z_scores > threshold
        print(f"Outliers detected: {outliers.sum()} / {len(outliers)}")
        
        return ~outliers, np.where(outliers)[0]
    
    def find_optimal_k(self, feature_matrix, k_range=range(2, 11)):
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Parameters:
        -----------
        feature_matrix : ndarray
            Preprocessed feature matrix
        k_range : range
            Range of k values to test
            
        Returns:
        --------
        Dictionary with metrics for each k value
        """
        print("\n=== Finding Optimal K ===")
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, 
                                n_init=10, max_iter=300)
            labels = kmeans_temp.fit_predict(feature_matrix)
            
            inertias.append(kmeans_temp.inertia_)
            silhouette_scores.append(silhouette_score(feature_matrix, labels))
            davies_bouldin_scores.append(davies_bouldin_score(feature_matrix, labels))
            calinski_harabasz_scores.append(calinski_harabasz_score(feature_matrix, labels))
            
            print(f"k={k}: Inertia={kmeans_temp.inertia_:.2f}, "
                  f"Silhouette={silhouette_scores[-1]:.3f}, "
                  f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Elbow plot
        axes[0, 0].plot(k_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True)
        
        # Silhouette score
        axes[0, 1].plot(k_range, silhouette_scores, 'go-')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score (Higher is Better)')
        axes[0, 1].grid(True)
        
        # Davies-Bouldin Index
        axes[1, 0].plot(k_range, davies_bouldin_scores, 'ro-')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Davies-Bouldin Index')
        axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[1, 0].grid(True)
        
        # Calinski-Harabasz Score
        axes[1, 1].plot(k_range, calinski_harabasz_scores, 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Calinski-Harabasz Score')
        axes[1, 1].set_title('Calinski-Harabasz Score (Higher is Better)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_k_analysis.png', dpi=300, bbox_inches='tight')
        print("\nOptimal k analysis plot saved to 'optimal_k_analysis.png'")
        
        return {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_harabasz_scores': calinski_harabasz_scores
        }
    
    def fit(self, feature_matrix):
        """
        Fit K-Means clustering model.
        
        Parameters:
        -----------
        feature_matrix : ndarray
            Preprocessed feature matrix
        """
        print(f"\n=== Fitting K-Means (k={self.n_clusters}) ===")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
            algorithm='lloyd'
        )
        
        self.cluster_labels = self.kmeans.fit_predict(feature_matrix)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        print(f"Clustering complete. Inertia: {self.kmeans.inertia_:.2f}")
        
        # Calculate metrics
        self.metrics['silhouette'] = silhouette_score(feature_matrix, self.cluster_labels)
        self.metrics['davies_bouldin'] = davies_bouldin_score(feature_matrix, self.cluster_labels)
        self.metrics['calinski_harabasz'] = calinski_harabasz_score(feature_matrix, self.cluster_labels)
        self.metrics['inertia'] = self.kmeans.inertia_
        
        print(f"\nClustering Metrics:")
        print(f"  Silhouette Score: {self.metrics['silhouette']:.3f}")
        print(f"  Davies-Bouldin Index: {self.metrics['davies_bouldin']:.3f}")
        print(f"  Calinski-Harabasz Score: {self.metrics['calinski_harabasz']:.1f}")
        
        # Distribution of laps across clusters
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"\nCluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} laps ({count/len(self.cluster_labels)*100:.1f}%)")
    
    def analyze_clusters(self, feature_matrix, metadata_df):
        """
        Analyze cluster characteristics.
        
        Parameters:
        -----------
        feature_matrix : ndarray
            Preprocessed feature matrix
        metadata_df : DataFrame
            Metadata for each lap
            
        Returns:
        --------
        DataFrame with cluster statistics
        """
        print("\n=== Analyzing Clusters ===")
        
        # Add cluster labels to metadata
        metadata_df['cluster'] = self.cluster_labels
        
        # Calculate statistics per cluster
        cluster_stats = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = feature_matrix[cluster_mask]
            cluster_meta = metadata_df[metadata_df['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'n_laps': cluster_mask.sum(),
                'mean_throttle_value': cluster_meta['mean_throttle'].mean(),
                'std_throttle_value': cluster_meta['mean_throttle'].std(),
                'max_throttle_value': cluster_meta['max_throttle'].mean(),
                'feature_mean': cluster_data.mean(),
                'feature_std': cluster_data.std()
            }
            
            cluster_stats.append(stats)
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Laps: {stats['n_laps']}")
            print(f"  Avg Throttle: {stats['mean_throttle_value']:.2f}%")
            print(f"  Throttle Variability: {stats['std_throttle_value']:.2f}")
        
        return pd.DataFrame(cluster_stats)
    
    def visualize_clusters(self, feature_matrix, metadata_df, sample_size=100):
        """
        Visualize clustering results.
        
        Parameters:
        -----------
        feature_matrix : ndarray
            Preprocessed feature matrix
        metadata_df : DataFrame
            Metadata for each lap
        sample_size : int
            Number of sample points for distance-based visualization
        """
        print("\n=== Generating Visualizations ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        axes[0, 0].bar(unique, counts, color=plt.cm.tab10(unique))
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Laps')
        axes[0, 0].set_title('Distribution of Laps Across Clusters')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Sample throttle traces by cluster
        sample_indices = np.random.choice(len(feature_matrix), 
                                         min(sample_size, len(feature_matrix)), 
                                         replace=False)
        
        for idx in sample_indices[:50]:  # Plot first 50 samples
            cluster_id = self.cluster_labels[idx]
            axes[0, 1].plot(feature_matrix[idx], alpha=0.3, 
                          color=plt.cm.tab10(cluster_id), linewidth=0.5)
        
        axes[0, 1].set_xlabel('Distance Point')
        axes[0, 1].set_ylabel('Normalized Throttle')
        axes[0, 1].set_title('Sample Throttle Traces by Cluster')
        
        # 3. Cluster centers
        for cluster_id in range(self.n_clusters):
            axes[1, 0].plot(self.cluster_centers[cluster_id], 
                          label=f'Cluster {cluster_id}',
                          linewidth=2)
        
        axes[1, 0].set_xlabel('Distance Point')
        axes[1, 0].set_ylabel('Normalized Throttle')
        axes[1, 0].set_title('Cluster Centers (Centroid Throttle Patterns)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Box plot of mean throttle by cluster
        metadata_df['cluster'] = self.cluster_labels
        cluster_throttle_data = [metadata_df[metadata_df['cluster'] == i]['mean_throttle'].values 
                                for i in range(self.n_clusters)]
        
        bp = axes[1, 1].boxplot(cluster_throttle_data, labels=range(self.n_clusters),
                               patch_artist=True)
        
        for patch, color in zip(bp['boxes'], plt.cm.tab10(range(self.n_clusters))):
            patch.set_facecolor(color)
        
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Mean Throttle (%)')
        axes[1, 1].set_title('Throttle Distribution by Cluster')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        print("Cluster visualization saved to 'cluster_visualization.png'")
    
    def calculate_driver_profiles(self, metadata_df, driver_column='driver_id'):
        """
        Calculate probabilistic driver style profiles.
        
        Parameters:
        -----------
        metadata_df : DataFrame
            Metadata with cluster assignments
        driver_column : str
            Column identifying drivers
            
        Returns:
        --------
        DataFrame with driver style distributions
        """
        print("\n=== Calculating Driver Profiles ===")
        
        metadata_df['cluster'] = self.cluster_labels
        
        # Group by driver
        driver_profiles = []
        
        if driver_column in metadata_df.columns:
            for driver_id in metadata_df[driver_column].unique():
                driver_laps = metadata_df[metadata_df[driver_column] == driver_id]
                
                # Calculate cluster distribution
                cluster_counts = driver_laps['cluster'].value_counts()
                total_laps = len(driver_laps)
                
                profile = {'driver_id': driver_id, 'total_laps': total_laps}
                
                for cluster_id in range(self.n_clusters):
                    percentage = (cluster_counts.get(cluster_id, 0) / total_laps) * 100
                    profile[f'style_{cluster_id}'] = percentage
                
                driver_profiles.append(profile)
                
                print(f"\nDriver {driver_id}:")
                print(f"  Total laps: {total_laps}")
                for cluster_id in range(self.n_clusters):
                    print(f"  Style {cluster_id}: {profile[f'style_{cluster_id}']:.1f}%")
        
        return pd.DataFrame(driver_profiles)
    
    def save_results(self, output_dir='/home/claude'):
        """
        Save clustering results to files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        print(f"\n=== Saving Results to {output_dir} ===")
        
        # Save cluster labels
        np.save(f'{output_dir}/cluster_labels.npy', self.cluster_labels)
        
        # Save cluster centers
        np.save(f'{output_dir}/cluster_centers.npy', self.cluster_centers)
        
        # Save metrics
        with open(f'{output_dir}/clustering_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print("Results saved successfully")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("F1 Telemetry K-Means Clustering Pipeline")
    print("=" * 60)
    
    # Initialize clustering
    clusterer = F1TelemetryKMeans(n_clusters=5, random_state=42)
    
    # Example with synthetic data (replace with your actual data)
    print("\n[NOTE] Replace this section with your actual data loading")
    print("Expected format: DataFrame with columns:")
    print("  - lap_id: unique identifier for each lap")
    print("  - distance: distance along track (meters)")
    print("  - throttle: throttle percentage (0-100)")
    print("  - driver_id: driver identifier (optional)")
    
    # Synthetic data example
    np.random.seed(42)
    n_laps = 200
    n_points_per_lap = 500
    
    synthetic_data = []
    for lap in range(n_laps):
        driver = f"DRV{lap % 10}"
        for point in range(n_points_per_lap):
            synthetic_data.append({
                'lap_id': lap,
                'driver_id': driver,
                'distance': point * 10,
                'throttle': np.random.normal(70, 15) + np.sin(point/50) * 20
            })
    
    df = pd.DataFrame(synthetic_data)
    
    # Preprocess data
    feature_matrix, lap_ids, metadata = clusterer.preprocess_data(
        df, 
        telemetry_column='throttle',
        distance_column='distance',
        lap_identifier='lap_id'
    )
    
    # Add driver info to metadata
    metadata['driver_id'] = [df[df['lap_id'] == lap_id]['driver_id'].iloc[0] 
                            for lap_id in lap_ids]
    
    # Detect outliers
    clean_mask, outlier_indices = clusterer.detect_outliers(feature_matrix)
    feature_matrix_clean = feature_matrix[clean_mask]
    metadata_clean = metadata[clean_mask].reset_index(drop=True)
    
    # Find optimal k
    k_metrics = clusterer.find_optimal_k(feature_matrix_clean, k_range=range(2, 11))
    
    # Fit clustering
    clusterer.fit(feature_matrix_clean)
    
    # Analyze clusters
    cluster_stats = clusterer.analyze_clusters(feature_matrix_clean, metadata_clean)
    
    # Visualize
    clusterer.visualize_clusters(feature_matrix_clean, metadata_clean)
    
    # Calculate driver profiles
    driver_profiles = clusterer.calculate_driver_profiles(metadata_clean, 'driver_id')
    
    # Save results
    clusterer.save_results()
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
