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

def create_dataset(windows, batch_size):
    """
    Converts a list of window data into an optimized TensorFlow Dataset.
    
    Args:
        windows: List of tuples (timestamp, temporal_context, weather_context, ridership_vector, y_true)
        batch_size: Size of batches to create
        
    Returns:
        A TensorFlow Dataset object optimized for GPU training
    """
    def generator():
        for window in windows:
            yield window
    
    # Define the output signature based on your data structure
    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),                # timestamp
        tf.TensorSpec(shape=(None, 24, 2), dtype=tf.float32),    # temporal_context [N,24,2]
        tf.TensorSpec(shape=(24, None), dtype=tf.float32),       # weather_context [24,F_ext]
        tf.TensorSpec(shape=(None,), dtype=tf.float32),          # ridership_vector
        tf.TensorSpec(shape=(None,), dtype=tf.float32)           # y_true [N]
    )
    
    # Create the dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    # Optimize the dataset for performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch while GPU is processing
    
    return dataset

@tf.function
def train_step(model, spatial_features, temporal_context, ridership_vector, weather_context, A, y_true):
    """
    Single optimization step compiled with tf.function for GPU acceleration.
    
    Args:
        model: The DSTGCN model
        spatial_features: Static spatial features for nodes
        temporal_context: Temporal features time series
        ridership_vector: External ridership features
        weather_context: Weather time series
        A: Adjacency matrix
        y_true: Ground truth values
        
    Returns:
        loss: The computed loss value
    """
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model((
            spatial_features, 
            temporal_context, 
            ridership_vector, 
            weather_context, 
            A
        ), training=True)
        
        # Calculate loss
        loss = model.loss(y_true, y_pred)
    
    # Calculate gradients and apply updates
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss

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

def make_windows(timestamps):
    windows = []
    timestamps = timestamps[:100]
    for ts in tqdm(timestamps, desc="Making windows"):
        temp_np    = np.stack([station_windows[sid][ts] for sid in station_ids])  # (N,24,2)
        weather_np = external_windows[ts]                                         # (24,F_ext)
        y_true_np  = y_true_df.loc[ts, station_ids].values.astype(np.float32)     # (N,)
        
        day_before = ts - pd.Timedelta(days=1)

        mask = (ridership_features.index.year == day_before.year) & \
        (ridership_features.index.month == day_before.month) & \
        (ridership_features.index.day == day_before.day)
        
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
    """
    Train the DSTGCN model using GPU-optimized data pipeline.
    
    Args:
        model: The DSTGCN model instance
        epochs: Number of training epochs
        batch_size: Batch size for training
        data: Tuple of (spatial_features, temporal_features, ridership_features, weather_features, A)
    """
    # Unpack data components
    spatial_features, temporal_features, ridership_features, weather_features, A = data
    
    # Convert spatial features to tensor once
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    
    # Filter timestamps
    timestamps = weather_features.index
    timestamps = timestamps[(timestamps.month > 1) | (timestamps.day > 1)]
    
    # Create windows once
    print("Creating data windows...")
    windows = make_windows(timestamps)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle windows each epoch
        random.shuffle(windows)
        
        # Create a fresh dataset with the shuffled windows
        print(f"Creating optimized dataset for epoch {epoch+1}...")
        dataset = create_dataset(windows, batch_size)
        
        epoch_loss = 0.0
        batch_count = 0
        
        # Train on batches using the dataset
        for batch_data in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            ts_batch, temporal_context_batch, weather_context_batch, ridership_batch, y_true_batch = batch_data
            
            batch_size_actual = tf.shape(ts_batch)[0]
            batch_loss = 0.0
            
            # Process each item in the batch
            for i in range(batch_size_actual):
                # Extract individual sample from batch
                temporal_context = temporal_context_batch[i]
                weather_context = tf.expand_dims(weather_context_batch[i], axis=0)
                ridership_vector = ridership_batch[i]
                y_true = tf.expand_dims(y_true_batch[i], axis=-1)
                
                # Use the optimized training step function
                loss = train_step(
                    model,
                    spatial_features,
                    temporal_context,
                    ridership_vector,
                    weather_context,
                    A,
                    y_true
                )
                
                batch_loss += loss
            
            # Average loss for this batch
            batch_loss /= tf.cast(batch_size_actual, tf.float32)
            
            batch_count += 1
            epoch_loss += batch_loss
        
        # Report average loss for the epoch
        average_epoch_loss = epoch_loss / tf.cast(batch_count, tf.float32)
        print(f"Average loss for epoch {epoch+1}: {average_epoch_loss}")