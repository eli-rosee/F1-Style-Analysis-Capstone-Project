"""
Quick Start Script for K-Means Clustering on TracingInsights Data
Run this after you've extracted telemetry JSON files from TracingInsights
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add parent directory to path to import clustering module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustering.kmeans_clustering import F1TelemetryKMeans


def load_scraped_telemetry_data(telemetry_dir='./telemetry', race_name=None):
    """
    Load telemetry data from YOUR scrape.py output structure.
    
    Expected structure (from your scrape.py):
    telemetry/
        Monaco_Grand_Prix/
            VER/
                1_tel.json
                2_tel.json
                ...
            HAM/
                1_tel.json
                ...
    
    Parameters:
    -----------
    telemetry_dir : str
        Path to the telemetry folder (default: './telemetry')
    race_name : str
        Specific race folder name (e.g., 'Monaco_Grand_Prix'), or None for all races
    
    Returns:
    --------
    DataFrame with all telemetry data
    """
    all_data = []
    
    # Check if telemetry directory exists
    if not os.path.exists(telemetry_dir):
        raise FileNotFoundError(
            f"Telemetry directory not found: {telemetry_dir}\n"
            f"Have you run data_ingestion/scrape.py yet?"
        )
    
    # List all race folders
    race_folders = [f for f in os.listdir(telemetry_dir) 
                   if os.path.isdir(os.path.join(telemetry_dir, f))]
    
    if not race_folders:
        raise ValueError(f"No race folders found in {telemetry_dir}")
    
    print(f"\nFound {len(race_folders)} race(s) in {telemetry_dir}:")
    for rf in race_folders:
        print(f"  - {rf}")
    
    # Filter by race if specified
    if race_name:
        if race_name not in race_folders:
            print(f"\nAvailable races: {race_folders}")
            raise ValueError(f"Race '{race_name}' not found in {telemetry_dir}")
        race_folders = [race_name]
    
    print(f"\nLoading data from {len(race_folders)} race(s)...")
    
    # Iterate through races
    for race_folder in race_folders:
        race_path = os.path.join(telemetry_dir, race_folder)
        
        # List all driver folders in this race
        driver_folders = [d for d in os.listdir(race_path) 
                         if os.path.isdir(os.path.join(race_path, d))]
        
        print(f"\n  {race_folder}: {len(driver_folders)} drivers")
        
        # Iterate through drivers
        for driver_code in driver_folders:
            driver_path = os.path.join(race_path, driver_code)
            
            # List all JSON files for this driver
            json_files = [f for f in os.listdir(driver_path) 
                         if f.endswith('_tel.json')]
            
            print(f"    - {driver_code}: {len(json_files)} laps")
            
            # Load each lap
            for filename in json_files:
                filepath = os.path.join(driver_path, filename)
                
                # Extract lap number from filename (e.g., "1_tel.json" -> 1)
                lap_number = int(filename.replace('_tel.json', ''))
                
                try:
                    with open(filepath, 'r') as f:
                        lap_data = json.load(f)
                    
                    # Handle the {"tel": {...}} wrapper
                    if 'tel' in lap_data:
                        tel_data = lap_data['tel']
                    else:
                        tel_data = lap_data

                    # Convert dict of arrays to DataFrame
                    df_lap = pd.DataFrame(tel_data)
                    
                    # Add metadata
                    df_lap['driver_id'] = driver_code
                    df_lap['lap_number'] = lap_number
                    df_lap['race_name'] = race_folder
                    
                    # Create unique lap identifier
                    df_lap['lap_id'] = f"{race_folder}_{driver_code}_lap{lap_number}"
                    
                    all_data.append(df_lap)
                    
                except Exception as e:
                    print(f"      Error loading {filepath}: {e}")
                    continue
    
    if not all_data:
        raise ValueError("No valid telemetry data found")
    
    # Combine all laps
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Loaded Telemetry Data Summary")
    print(f"{'='*60}")
    print(f"Total records: {len(df_combined):,}")
    print(f"Unique laps: {df_combined['lap_id'].nunique()}")
    print(f"Drivers: {sorted(df_combined['driver_id'].unique())}")
    print(f"Races: {sorted(df_combined['race_name'].unique())}")
    print(f"{'='*60}\n")
    
    return df_combined


def load_tracing_insights_data(data_directory, race_name=None, session_type='Race'):
    """
    Load TracingInsights JSON files into a single DataFrame.
    
    TracingInsights structure:
    data_directory/
        2025/
            Canada/
                Race/
                    VER_lap_1.json
                    VER_lap_2.json
                    HAM_lap_1.json
                    ...
    
    Parameters:
    -----------
    data_directory : str
        Path to TracingInsights data folder
    race_name : str
        Specific race to load (e.g., 'Monaco'). If None, loads all races.
    session_type : str
        'Race', 'Q1', 'Q2', 'Q3', etc.
    
    Returns:
    --------
    DataFrame with all telemetry data
    """
    all_data = []
    
    # Walk through directory structure
    for year_folder in os.listdir(data_directory):
        year_path = os.path.join(data_directory, year_folder)
        
        if not os.path.isdir(year_path):
            continue
            
        for race_folder in os.listdir(year_path):
            # Filter by race if specified
            if race_name and race_folder != race_name:
                continue
                
            race_path = os.path.join(year_path, race_folder)
            session_path = os.path.join(race_path, session_type)
            
            if not os.path.exists(session_path):
                continue
            
            # Load all JSON files in this session
            for filename in os.listdir(session_path):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(session_path, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        lap_data = json.load(f)
                    
                    # Extract driver and lap number from filename
                    # Format: DRIVER_lap_NUMBER.json
                    parts = filename.replace('.json', '').split('_')
                    driver_code = parts[0]
                    lap_number = int(parts[2])
                    
                    # Convert telemetry to DataFrame
                    df_lap = pd.DataFrame(lap_data)
                    
                    # Add metadata
                    df_lap['driver_id'] = driver_code
                    df_lap['lap_number'] = lap_number
                    df_lap['race_name'] = race_folder
                    df_lap['year'] = year_folder
                    df_lap['session_type'] = session_type
                    
                    # Create unique lap identifier
                    df_lap['lap_id'] = f"{year_folder}_{race_folder}_{driver_code}_lap{lap_number}"
                    
                    all_data.append(df_lap)
                    
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue
    
    if not all_data:
        raise ValueError(f"No data found in {data_directory}")
    
    # Combine all laps
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Loaded TracingInsights Data")
    print(f"{'='*60}")
    print(f"Total records: {len(df_combined):,}")
    print(f"Unique laps: {df_combined['lap_id'].nunique()}")
    print(f"Drivers: {df_combined['driver_id'].nunique()}")
    print(f"Races: {df_combined['race_name'].nunique()}")
    print(f"Years: {df_combined['year'].nunique()}")
    
    return df_combined


def run_clustering_analysis(df, n_clusters=5, output_dir='./clustering_results'):
    """
    Run complete K-Means clustering analysis on telemetry data.
    
    Parameters:
    -----------
    df : DataFrame
        Telemetry data loaded from TracingInsights
    n_clusters : int
        Number of clusters for K-Means
    output_dir : str
        Directory to save results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"K-Means Clustering Analysis")
    print(f"{'='*60}")
    
    # Initialize clusterer
    clusterer = F1TelemetryKMeans(n_clusters=n_clusters, random_state=42)
    
    # Preprocess data
    print("\n[1/10] Preprocessing data...")
    feature_matrix, lap_ids, metadata = clusterer.preprocess_data(
        df,
        telemetry_column='throttle',
        distance_column='distance',
        lap_identifier='lap_id'
    )
    
    # Add driver information to metadata
    print("[2/10] Adding metadata...")
    metadata['driver_id'] = [
        df[df['lap_id'] == lap_id]['driver_id'].iloc[0] 
        for lap_id in lap_ids
    ]
    metadata['race_name'] = [
        df[df['lap_id'] == lap_id]['race_name'].iloc[0] 
        for lap_id in lap_ids
    ]
    
    # Detect outliers
    print("[3/10] Detecting outliers...")
    clean_mask, outlier_indices = clusterer.detect_outliers(feature_matrix, threshold=3)
    feature_matrix_clean = feature_matrix[clean_mask]
    metadata_clean = metadata[clean_mask].reset_index(drop=True)
    
    print(f"  Removed {len(outlier_indices)} outlier laps")
    print(f"  Clean dataset: {len(feature_matrix_clean)} laps")
    
    # Find optimal k
    print("[4/10] Finding optimal number of clusters...")
    k_metrics = clusterer.find_optimal_k(
        feature_matrix_clean, 
        k_range=range(2, 11)
    )
    
    # Save optimal k plot
    if os.path.exists('/home/claude/optimal_k_analysis.png'):
        os.rename(
            '/home/claude/optimal_k_analysis.png',
            os.path.join(output_dir, 'optimal_k_analysis.png')
        )
    
    # Fit K-Means
    print(f"[5/10] Fitting K-Means with k={n_clusters}...")
    clusterer.fit(feature_matrix_clean)
    
    # Analyze clusters
    print("[6/10] Analyzing cluster characteristics...")
    cluster_stats = clusterer.analyze_clusters(feature_matrix_clean, metadata_clean)
    cluster_stats.to_csv(os.path.join(output_dir, 'cluster_statistics.csv'), index=False)
    
    # Visualize clusters
    print("[7/10] Generating visualizations...")
    clusterer.visualize_clusters(feature_matrix_clean, metadata_clean)
    if os.path.exists('/home/claude/cluster_visualization.png'):
        os.rename(
            '/home/claude/cluster_visualization.png',
            os.path.join(output_dir, 'cluster_visualization.png')
        )
    
    # Calculate driver profiles
    print("[8/10] Calculating driver profiles...")
    driver_profiles = clusterer.calculate_driver_profiles(metadata_clean, 'driver_id')
    driver_profiles.to_csv(os.path.join(output_dir, 'driver_profiles.csv'), index=False)
    
    # Save raw results
    print("[9/10] Saving results...")
    clusterer.save_results(output_dir=output_dir)
    
    # Create summary report
    print("[10/10] Generating summary report...")
    with open(os.path.join(output_dir, 'clustering_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("K-Means Clustering Summary Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"  Total laps analyzed: {len(feature_matrix_clean)}\n")
        f.write(f"  Drivers: {metadata_clean['driver_id'].nunique()}\n")
        f.write(f"  Races: {metadata_clean['race_name'].nunique()}\n")
        f.write(f"  Outliers removed: {len(outlier_indices)}\n\n")
        
        f.write(f"Clustering Configuration:\n")
        f.write(f"  Number of clusters (k): {n_clusters}\n")
        f.write(f"  Random seed: 42\n")
        f.write(f"  Algorithm: K-Means (Lloyd)\n\n")
        
        f.write(f"Clustering Quality Metrics:\n")
        f.write(f"  Silhouette Score: {clusterer.metrics['silhouette']:.4f}\n")
        f.write(f"  Davies-Bouldin Index: {clusterer.metrics['davies_bouldin']:.4f}\n")
        f.write(f"  Calinski-Harabasz Score: {clusterer.metrics['calinski_harabasz']:.2f}\n")
        f.write(f"  Inertia: {clusterer.metrics['inertia']:.2f}\n\n")
        
        f.write(f"Cluster Distribution:\n")
        unique, counts = np.unique(clusterer.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(clusterer.cluster_labels)) * 100
            f.write(f"  Cluster {cluster_id}: {count} laps ({percentage:.1f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Files Generated:\n")
        f.write("  - optimal_k_analysis.png: Metrics for choosing k\n")
        f.write("  - cluster_visualization.png: Visual analysis of clusters\n")
        f.write("  - cluster_statistics.csv: Statistics per cluster\n")
        f.write("  - driver_profiles.csv: Driver style distributions\n")
        f.write("  - cluster_labels.npy: Cluster assignments\n")
        f.write("  - cluster_centers.npy: Cluster centroids\n")
        f.write("  - clustering_metrics.json: Quality metrics\n")
        f.write("="*60 + "\n")
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey files:")
    print(f" cluster_visualization.png - Visual analysis")
    print(f" optimal_k_analysis.png - Choosing optimal k")
    print(f" driver_profiles.csv - Driver style distributions")
    print(f" clustering_summary.txt - Full report")
    
    return clusterer, driver_profiles, cluster_stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("TracingInsights K-Means Clustering - Quick Start")
    print("="*60)
    
    # ========================================
    # CONFIGURATION - EDIT THESE VALUES
    # ========================================
    
    # OPTION 1: Use data from YOUR scrape.py script (RECOMMENDED)
    # This uses the telemetry/ folder created by data_ingestion/scrape.py
    USE_SCRAPED_DATA = True  # Set to False if using raw TracingInsights repo
    TELEMETRY_DIR = "./telemetry"  # Path to telemetry folder from scrape.py
    
    # OPTION 2: Use raw TracingInsights GitHub structure (if you cloned it)
    # Path to TracingInsights data folder
    DATA_DIRECTORY = "/path/to/TracingInsights/data"
    SESSION_TYPE = "Race"  # or "Q1", "Q2", "Q3"
    
    # COMMON SETTINGS
    # Optional: Filter by specific race (set to None for all races)
    RACE_NAME = "Canadian_Grand_Prix"  # e.g., "Monaco_Grand_Prix" (with underscores!) or None
    
    # Number of clusters for K-Means
    N_CLUSTERS = 5
    
    # Output directory for results
    OUTPUT_DIR = "./clustering_results"
    
    # ========================================
    # RUN ANALYSIS
    # ========================================
    
    try:
        # Load data
        print("\n[STEP 1] Loading telemetry data...")
        
        if USE_SCRAPED_DATA:
            # Use data from your scrape.py
            print(f"Loading from scrape.py output: {TELEMETRY_DIR}")
            df = load_scraped_telemetry_data(
                telemetry_dir=TELEMETRY_DIR,
                race_name=RACE_NAME
            )
        else:
            # Use raw TracingInsights structure
            print(f"Loading from TracingInsights repo: {DATA_DIRECTORY}")
            df = load_tracing_insights_data(
                DATA_DIRECTORY, 
                race_name=RACE_NAME,
                session_type=SESSION_TYPE
            )
        
        # Run clustering
        print("\n[STEP 2] Running clustering analysis...")
        clusterer, driver_profiles, cluster_stats = run_clustering_analysis(
            df, 
            n_clusters=N_CLUSTERS,
            output_dir=OUTPUT_DIR
        )
        
        # Display top results
        print("\n" + "="*60)
        print("Driver Profile Summary (Top 5 Drivers)")
        print("="*60)
        print(driver_profiles.head())
        
        print("\n" + "="*60)
        print("Cluster Statistics")
        print("="*60)
        print(cluster_stats)
        
        print("\n SUCCESS! Check the output directory for detailed results.")
        
    except FileNotFoundError as e:
        print("\n ERROR: Data not found!")
        print(str(e))
        print("\n SOLUTION:")
        if USE_SCRAPED_DATA:
            print(f"1. Run data_ingestion/scrape.py to download telemetry data")
            print(f"2. Check that the telemetry/ folder exists")
            print(f"3. Current TELEMETRY_DIR: {TELEMETRY_DIR}")
        else:
            print(f"1. Update DATA_DIRECTORY to point to your TracingInsights data")
            print(f"2. Current path: {DATA_DIRECTORY}")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
