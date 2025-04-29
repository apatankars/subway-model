import pandas as pd
import json
import pickle

with open("wjsons/2023.json", 'r') as file:
    w1 = json.load(file)

with open("wjsons/2024.json", 'r') as file:
    w2 = json.load(file)

with open("wjsons/2025.json", 'r') as file:
    w3 = json.load(file)

w23 = pd.DataFrame(w1)
w24 = pd.DataFrame(w2)
w25 = pd.DataFrame(w3)

w = pd.concat([w23, w24, w25], ignore_index=True)
w = w[['date','hour','temp','dew_point', 'humidity','precipitation', 'wind_speed', 'pressure']]

with open('weather_pandas.pkl', 'wb') as f:  
    pickle.dump(w, f)
      

