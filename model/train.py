import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import os
import random
import pickle
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.training_prep import get_turnstile_context, get_external_context
from numpy.lib.stride_tricks import sliding_window_view

def split_external(external_features: pd.DataFrame):
    external_features = external_features.set_index('datetime').copy()
    
    weather_features = external_features[['temp','dew_point','humidity','precipitation','wind_speed','pressure',
                                          'hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7',
                                          'hour_8','hour_9','hour_10','hour_11','hour_12','hour_13','hour_14',
                                          'hour_15','hour_16','hour_17','hour_18','hour_19','hour_20','hour_21',
                                          'hour_22','hour_23','month_1','month_2','month_3','month_4','month_5',
                                          'month_6','month_7','month_8','month_9','month_10','month_11','month_12']]
    
    ridership_features = external_features[['0','1','2','3','4','5','6','subways_ridership','subways_percent_of_pre',
                                            'buses_ridership','buses_percent_of_pre','lirr_ridership','lirr_percent_of_pre',
                                            'mn_ridership','mn_percent_of_pre','access_a_ride_trips',
                                            'access_a_ride_percent_of_pre','bridges_tunnels_traffic',
                                            'bridges_tunnels_percent_of_pre','sir_ridership','sir_percent_of_pre']]
    
    ridership_features = ridership_features[ridership_features["subways_percent_of_pre"] != 0]
    
    return ridership_features, weather_features

with open("data/subway_network.pkl", "rb") as f:
    graph = pickle.load(f)

temporal_features = pd.read_parquet("data/final_data/temporal_2023.parquet")
external_features = pd.read_parquet("data/final_data/external_2023.parquet")
ridership_features, weather_features = split_external(external_features)

tf_df = temporal_features.copy()
if not isinstance(tf_df.index, pd.DatetimeIndex):
    tf_df.set_index('transit_timestamp', inplace=True)
tf_df.sort_index(inplace=True)

station_ids = [int(sid) for sid in graph.nodes]

station_windows = {}
for sid, grp in tf_df.groupby('station_complex_id'):
    arr = grp[['ridership','transfers']].values       
    wins = sliding_window_view(arr, window_shape=24, axis=0)  
    ts_ends = grp.index[24:]                             
    station_windows[sid] = dict(zip(ts_ends, wins))

weather_arr = weather_features.values                   
ext_wins   = sliding_window_view(weather_arr, 24, axis=0)  
ext_ts     = weather_features.index[23:]
external_windows = dict(zip(ext_ts, ext_wins))

y_true_df = (
    tf_df
    .reset_index()
    .pivot(
        index='transit_timestamp',
        columns='station_complex_id',
        values='ridership'
    )
    .reindex(columns=station_ids, fill_value=0)
)

# def make_windows(timestamps, temporal_features, weather_features):
#     weather_list = []
#     temporal_list = []
#     y_true_list = []
    
#     for ts in tqdm(timestamps, desc="Making windows"):
#         if (ts.day == 2 and ts.month == 1):
#             print("did 1")
#         ts_list = []
#         y_true_per_ts = []  # collect all y_true for this timestamp
        
#         weather_list.append(
#             tf.convert_to_tensor(get_external_context(ts, weather_features, 24), 
#                                  dtype=tf.float32)
#         )
        
#         for node_id in graph.nodes:
#             tens = tf.convert_to_tensor(get_turnstile_context(ts, int(node_id), temporal_features, 24),
#                                                 dtype=tf.float32)
#             if len(tens) != 24:
#                 print(tf.shape(tens))
#             ts_list.append(tens)
            
#             match = temporal_features[
#                 (temporal_features["transit_timestamp"] == ts) & 
#                 (temporal_features["station_complex_id"] == int(node_id))
#             ]
#             if not match.empty:
#                 y_true_value = match["ridership"].values[0] 
#             else:
#                 y_true_value = 0.0  
            
#             y_true_per_ts.append(y_true_value)
        
#         temporal_list.append(tf.convert_to_tensor(ts_list, dtype=tf.float32)) 
#         y_true_list.append(tf.convert_to_tensor(y_true_per_ts, dtype=tf.float32))
    
#     return list(zip(timestamps, temporal_list, weather_list, y_true_list))

def make_windows(timestamps):
    windows = []
    for ts in tqdm(timestamps, desc="Making windows"):
        temp_np    = np.stack([station_windows[sid][ts] for sid in station_ids])  # (N,24,2)
        weather_np = external_windows[ts]                                         # (24,F_ext)
        y_true_np  = y_true_df.loc[ts, station_ids].values.astype(np.float32)     # (N,)
        
        mask = (ridership_features.index.year == ts.year) & \
               (ridership_features.index.month == ts.month) & \
               (ridership_features.index.day == ts.day)
        arr = ridership_features.loc[mask].values
        ridership_vector = tf.convert_to_tensor(arr, dtype=tf.float32)
        
        windows.append((
            ts,
            tf.convert_to_tensor(temp_np,    dtype=tf.float32),  # [N,24,2]
            tf.convert_to_tensor(weather_np, dtype=tf.float32),  # [24,F_ext]
            ridership_vector,                                    # [F_rid, 1]
            tf.convert_to_tensor(y_true_np,  dtype=tf.float32)   # [N]
        ))
    return windows


def train(model, epochs, batch_size, data):
    # spatial_features, temporal_features, external_features, weather_features, A = model_inputs
    
    spatial_features, temporal_features, external_features, A = data
    ridership_features, weather_features = split_external(external_features)
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    
    timestamps = weather_features.index
    timestamps = timestamps[(timestamps.month > 1) | (timestamps.day > 1)]
    # print("\nmaking window")
    windows = make_windows(timestamps)
    
    for epoch in range(epochs):
        # shuffle timestamps each epoch
        random.shuffle(windows)
            
        # "batching"
        for i in tqdm(range(0, len(windows), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = windows[i:i + batch_size]

            with tf.GradientTape() as tape:
                total_loss = 0.0
                for (ts, temporal_context, weather_context, ridership_vector, y_true) in batch:
                    
                    weather_context = tf.expand_dims(weather_context, axis=0)
                    y_true = tf.expand_dims(y_true, axis=-1)
                    
                    # forward pass on one window
                    y_pred = model((spatial_features, temporal_context, ridership_vector, weather_context, A), training=True)  # [N,1]
                    total_loss += model.loss(y_true, y_pred)

                # average the loss over the K windows
                total_loss /= tf.cast(len(batch), tf.float32)

            # backprop once
            grads = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print(f"average loss for epoch {epoch+1}: {total_loss}")
    
    