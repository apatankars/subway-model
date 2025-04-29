import pandas as pd
import tensorflow as tf
import networkx as nx
import pickle
import torch

weather_data = pd.read_pickle("data/weather_pandas.pkl")
ridership_data = pd.read_pickle("data/transport_ridership.pkl")
with open("data/subway_network.pkl", "rb") as f:
    graph = pickle.load(f)

# Cleaning/normalizing turnstile data
turnstile_2023 = pd.read_parquet("data/turnstile_data/2023_turnstile_data.parquet")
turnstile_2024 = pd.read_parquet("data/turnstile_data/2024_turnstile_data.parquet")

scalers_2023 = turnstile_2023.groupby('station_complex_id').agg({
    'transfers': ['min', 'max'],
    'ridership': ['min', 'max'],
})
scalers_2023.columns = ['transfers_min', 'transfers_max', 'ridership_min', 'ridership_max']
scalers_2023 = scalers_2023.reset_index()

turnstile_2023 = turnstile_2023.merge(scalers_2023, on='station_complex_id', how='left')
epsilon = 1e-8
turnstile_2023['transfers'] = (turnstile_2023['transfers'] - turnstile_2023['transfers_min']) / (turnstile_2023['transfers_max'] - turnstile_2023['transfers_min'] + epsilon)
turnstile_2023['ridership'] = (turnstile_2023['ridership'] - turnstile_2023['ridership_min']) / (turnstile_2023['ridership_max'] - turnstile_2023['ridership_min'] + epsilon)

ridership_scalers = scalers_2023.set_index('station_complex_id')[['ridership_min', 'ridership_max']].to_dict(orient='index')
turnstile_2024 = turnstile_2024.merge(scalers_2023, on='station_complex_id', how='left')

turnstile_2024['transfers'] = (turnstile_2024['transfers'] - turnstile_2024['transfers_min']) / (turnstile_2024['transfers_max'] - turnstile_2024['transfers_min'] + epsilon)
turnstile_2024['ridership'] = (turnstile_2024['ridership'] - turnstile_2024['ridership_min']) / (turnstile_2024['ridership_max'] - turnstile_2024['ridership_min'] + epsilon)

# Add min/max from training to each node to convert back to numerical inputs to get meaningful predictions
for node_id in graph.nodes:
    scaler = ridership_scalers.get(int(node_id), None)
    graph.nodes[node_id]['ridership_min'] = scaler['ridership_min']
    graph.nodes[node_id]['ridership_max'] = scaler['ridership_max']

# Combine weather and ridership data to get external features
def edit_weather_data(weather_data):
    weather_data = weather_data.copy()
    weather_data['datetime'] = pd.to_datetime({
        'year': weather_data['year'],
        'month': weather_data['month'],
        'day': weather_data['day'],
        'hour': weather_data['hour'],
    })
    weather_data = weather_data.drop(columns=['year', 'day'])
    weather_data = pd.get_dummies(weather_data, columns=['hour', 'month'], prefix=['hour', 'month'])
    return weather_data

weather_2023 = weather_data[weather_data["year"] == 2023].copy()
weather_2024 = weather_data[weather_data["year"] == 2024].copy()
weather_2023 = edit_weather_data(weather_2023)
weather_2024 = edit_weather_data(weather_2024)

ridership_2023 = ridership_data[ridership_data["date"].dt.year == 2023]
ridership_2024 = ridership_data[ridership_data["date"].dt.year == 2024]

def make_external_features(weather_data, rider_data):
    weather_data = weather_data.copy()
    rider_data = rider_data.copy()
    weather_data['date'] = weather_data['datetime'].dt.date
    
    ridership_columns = ['subways_ridership', 'subways_percent_of_pre', 'buses_ridership', 
                         'buses_percent_of_pre', 'lirr_ridership', 'lirr_percent_of_pre',
                         'mn_ridership', 'mn_percent_of_pre', 'access_a_ride_trips', 'access_a_ride_percent_of_pre',
                         'bridges_tunnels_traffic', 'bridges_tunnels_percent_of_pre', 'sir_ridership',
                         'sir_percent_of_pre']

    for col in ridership_columns:
        weather_data[col] = 0.0

    mask_23 = weather_data['datetime'].dt.hour == 23

    for col in ridership_columns:
        weather_data.loc[mask_23, col] = weather_data.loc[mask_23, 'date'].map(
            rider_data.set_index('date')[col]
        )
    external_data = weather_data.drop(columns=['date']).set_index('datetime')
    return external_data


# Final data
turnstile_2023 = turnstile_2023.drop(columns=['transfers_min', 'transfers_max', 'ridership_min', 'ridership_max'])
turnstile_2024 = turnstile_2024.drop(columns=['transfers_min', 'transfers_max', 'ridership_min', 'ridership_max'])
external_2023 = make_external_features(weather_2023, ridership_2023)
external_2024 = make_external_features(weather_2024, ridership_2024)

def get_turnstile_context(timestamp: pd.Timestamp, station_id: int, turnstile_dataset: pd.DataFrame, context_hours: int):
    t = pd.Timestamp(timestamp)

    if not isinstance(turnstile_dataset.index, pd.DatetimeIndex):
        turnstile_dataset = turnstile_dataset.set_index('transit_timestamp')

    station_df = turnstile_dataset[turnstile_dataset['station_complex_id'] == station_id]

    start_t = t - pd.Timedelta(hours=context_hours)
    end_t = t - pd.Timedelta(hours=1)

    window = station_df.loc[start_t:end_t]
    assert len(window) == 24
    return window.sort_index()[['ridership', 'transfers']]

def get_external_context(timestamp: pd.Timestamp, external_dataset: pd.DataFrame, context_hours: int):
    t = pd.Timestamp(timestamp)

    if not isinstance(external_dataset.index, pd.DatetimeIndex):
        external_dataset = external_dataset.set_index('datetime')

    start_t = t - pd.Timedelta(hours=context_hours)
    end_t = t - pd.Timedelta(hours=1)

    window = external_dataset.loc[start_t:end_t]

    return window.sort_index()

timestamp = pd.Timestamp("2024-11-02 22:00:00")
assert (timestamp.day != 1)  or (timestamp.month != 1) 
station_id = 1
turnstile_data = turnstile_2024
external_data = external_2024
window = 24
# print(get_turnstile_context(timestamp, station_id, turnstile_data, window))
# print(get_external_context(timestamp, external_data, window))