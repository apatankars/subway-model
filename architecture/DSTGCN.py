import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import split_external, make_windows, create_dataset

@tf.function
def evaluate_step(model, spatial_features, temporal_context, ridership_vector, weather_context, A, y_true):
    """
    Single evaluation step compiled with tf.function for GPU acceleration.
    
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
    # Forward pass
    y_pred = model((
        spatial_features, 
        temporal_context, 
        ridership_vector, 
        weather_context, 
        A
    ), training=False)
    
    # Calculate loss
    loss = model.loss(y_true, y_pred)
    
    return loss

def test(model, batch_size, data):
    """
    Test the DSTGCN model using GPU-optimized data pipeline.
    
    Args:
        model: The DSTGCN model instance
        batch_size: Batch size for evaluation
        data: Tuple of (spatial_features, temporal_features, ridership_features, weather_features, A)
        
    Returns:
        average_batch_loss: Average loss across all test batches
    """
    # Unpack data components
    spatial_features, temporal_features, ridership_features, weather_features, A = data
    
    # Convert spatial features to tensor once
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    
    # Filter timestamps
    timestamps = weather_features.index
    timestamps = timestamps[(timestamps.month > 1) | (timestamps.day > 1)]
    
    # Create windows for testing
    print("Creating test data windows...")
    windows = make_windows(timestamps)
    
    # Create an optimized dataset for testing
    print("Creating optimized test dataset...")
    dataset = create_dataset(windows, batch_size)
    
    # Track test loss
    batch_losses = []
    
    # Evaluate on batches
    for batch_data in tqdm(dataset, desc="Testing"):
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
            
            # Use the optimized evaluation step function
            loss = evaluate_step(
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
        batch_losses.append(batch_loss)
    
    # Calculate overall average loss
    average_batch_loss = tf.reduce_mean(batch_losses)
    print(f"Test loss: {average_batch_loss.numpy()}")
    
    return average_batch_loss