import fastf1
import requests
import logging

base_url = 'https://github.com/TracingInsights-Archive/2025/tree/main/'
end_url = '/Race'

logging.disable(logging.INFO)

def main():
    session = fastf1.get_session(2025, 1, 'R')
    session.load()
    event_name = session.event['EventName']
    drivers_list = []

    for driver in session.drivers:
        drivers_list.append(session.get_driver(driver)['Abbreviation'])

    url_event_name = event_name.replace(' ', '%20')

    url = base_url + url_event_name + end_url
    
    print(url_event_name)
    print(drivers_list)
    print(url)

if __name__ == '__main__':
    main()