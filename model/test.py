import tensorflow as tf
from train import split_external, make_windows
from tqdm import tqdm

def test(model, batch_size, data):
    # spatial_features, temporal_features, external_features, weather_features, A = model_inputs
    
    spatial_features, temporal_features, external_features, A = data
    ridership_features, weather_features = split_external(external_features)
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    
    timestamps = weather_features.index
    timestamps = timestamps[(timestamps.month > 1) | (timestamps.day > 1)]
    windows = make_windows(timestamps)
        
    # "batching"
    batch_losses = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]

        batch_loss = 0.0
        for (ts, temporal_context, weather_context, ridership_vector, y_true) in batch:
            weather_context = tf.expand_dims(weather_context, axis=0)
            y_true = tf.expand_dims(y_true, axis=-1)
            # forward pass on one window
            y_pred = model(spatial_features, temporal_context, ridership_vector, weather_context, A, training=True)  # [N,1]
            batch_loss += model.loss(y_true, y_pred)

        # average the loss over the K windows
        batch_loss /= tf.cast(len(batch), tf.float32)
        batch_losses.append(batch_loss)
    
    average_batch_loss = tf.reduce_mean(batch_losses)
    print(f"Test loss: {average_batch_loss.numpy()}")
    return average_batch_loss
