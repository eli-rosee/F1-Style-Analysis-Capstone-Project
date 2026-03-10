import json
import os
import fastf1
import logging

# Disables logging messages from FastF1 API
logging.disable(logging.INFO)

YEAR = 2025
CACHE_FILE = 'race_metadata_cache.json'

DB_CODE_MAP = {
    'Australian_Grand_Prix':     'aus',
    'Chinese_Grand_Prix':        'chn',
    'Japanese_Grand_Prix':       'jpn',
    'Bahrain_Grand_Prix':        'bah',
    'Saudi_Arabian_Grand_Prix':  'sau',
    'Miami_Grand_Prix':          'mia',
    'Emilia_Romagna_Grand_Prix': 'emi',
    'Monaco_Grand_Prix':         'mon',
    'Spanish_Grand_Prix':        'esp',
    'Canadian_Grand_Prix':       'can',
    'Austrian_Grand_Prix':       'aut',
    'British_Grand_Prix':        'gbr',
    'Belgian_Grand_Prix':        'bel',
    'Hungarian_Grand_Prix':      'hun',
    'Dutch_Grand_Prix':          'ned',
    'Italian_Grand_Prix':        'ita',
    'Azerbaijan_Grand_Prix':     'aze',
    'Singapore_Grand_Prix':      'sin',
    'United_States_Grand_Prix':  'usa',
    'Mexico_City_Grand_Prix':    'mex',
    'São_Paulo_Grand_Prix':      'sao',
    'Las_Vegas_Grand_Prix':      'las',
    'Qatar_Grand_Prix':          'qat',
    'Abu_Dhabi_Grand_Prix':      'abu',
}


def main():
    if os.path.exists(CACHE_FILE):
        print("Cache already exists. Delete f1_cache.json to re-fetch.")
        return

    schedule = fastf1.get_event_schedule(YEAR)[['Country', 'EventName', 'RoundNumber']]
    schedule = schedule[~schedule['EventName'].str.contains('Testing', case=False, na=False)]

    cache = {'races': {}}

    for _, row in schedule.iterrows():
        race_name = row['EventName'].replace(' ', '_')
        round_num = int(row['RoundNumber'])
        db_code = DB_CODE_MAP.get(race_name)

        if not db_code:
            print(f"  WARNING: No db_code found for {race_name}, skipping.")
            continue

        print(f"Loading {race_name}...")
        try:
            session = fastf1.get_session(YEAR, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            driver_laps = session.laps.groupby('Driver')['LapNumber'].max().astype(int).to_dict()
        except Exception as e:
            print(f"  WARNING: {e}")
            driver_laps = {}

        cache['races'][race_name] = {
            'telemetry_db_code': f'telemetry_{db_code}_{YEAR}',
            'metadata_db_code': f'metadata_{db_code}_{YEAR}',
            'driver_laps': driver_laps
        }

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    print("Done.")


if __name__ == '__main__':
    main()
