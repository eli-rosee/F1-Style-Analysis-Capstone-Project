"""
Custom data loader for scrape.py output structure.
Compatible with your existing telemetry folder structure.
"""

import os
import json
import pandas as pd


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
                    
                    # Convert to DataFrame
                    df_lap = pd.DataFrame(lap_data)
                    
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


# Example usage
if __name__ == "__main__":
    # Load all races
    df = load_scraped_telemetry_data(telemetry_dir='./telemetry')
    
    # OR: Load specific race
    # df = load_scraped_telemetry_data(
    #     telemetry_dir='./telemetry', 
    #     race_name='Monaco_Grand_Prix'
    # )
    
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
