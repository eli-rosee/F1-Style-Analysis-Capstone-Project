import fastf1
import requests
import logging
from urllib.parse import quote
import os

base_url = 'https://raw.githubusercontent.com/TracingInsights-Archive/2025/main/'
end_url = '/Race'

file_extension = '_tel.json'

logging.disable(logging.INFO)

def main():
    session = fastf1.get_session(2025, 1, 'R')
    session.load()
    event_name = session.event['EventName']

    drivers_list = []
    driver_dict = {}
    
    laps_dict = {}

    for driver in session.drivers:
        driver_dict[driver] = session.get_driver(driver)['Abbreviation']
        drivers_list.append(session.get_driver(driver)['Abbreviation'])

    url_event_name = quote(event_name, safe='')
    url = base_url + url_event_name + end_url

    for driver in driver_dict.keys():
        laps_dict[driver] = int(session.results['Laps'][driver] + 0.99)

    for driver in driver_dict.keys():
        abbr = driver_dict[driver]

        for lap in range(1, laps_dict[driver] + 1):
            addition_url = f'/{abbr}/{lap}{file_extension}'
            download_url = url + addition_url

            print(abbr, lap)
            r = requests.get(download_url)
            if r.status_code == 200:
                path = f'telemetry/{event_name}/Race/{abbr}'
                os.makedirs(path, exist_ok=True)
                with open(f'{path}/{lap}{file_extension}', 'wb') as f:
                    f.write(r.content)


if __name__ == '__main__':
    main()
