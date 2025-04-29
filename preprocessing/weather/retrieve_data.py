import pandas as pd
import json 

w = pd.read_csv('2022.csv.gz', compression='gzip')
w = w.loc[:, ~w.columns.str.contains('^Unnamed')]

w.columns = [
    'date', 'hour', 'temp', 'dew_point', 'humidity',
    'precipitation', 'wind_direction', 'wind_speed',
    'pressure', 'weather_code'
]

json_data = w.to_dict(orient='records')

with open('2022.json', 'w') as f:
    json.dump(json_data, f)

