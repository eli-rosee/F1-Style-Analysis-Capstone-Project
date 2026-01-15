import fastf1
import requests
import logging

base_url = 'https://github.com/TracingInsights-Archive/2025/tree/main/'
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

    url_event_name = event_name.replace(' ', '%20')

    url = base_url + url_event_name + end_url
    
    for driver in driver_dict.keys():
        laps_dict[driver] = int(session.results['Laps'][driver] + 0.99)
    
    for driver in driver_dict.keys():
        
        addition_url = ''
        ## download url + addition_url + file_extension using requests
        pass
    
    print(laps_dict)

if __name__ == '__main__':
    main()