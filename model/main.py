import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from architecture.DSTGCN import DSTGCN  
from train import train
from preprocessing.build_adjacency_matrix import build_adjacency_matrix
from test import test
import time

def save_model_with_fallbacks(model, base_path="model/saved_models"):
    """Save model using multiple methods for redundancy"""
    import os
    import time
    import json
    
    # Create timestamp for unique folder
    timestamp = int(time.time())
    save_dir = f"{base_path}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Try saving full model
    try:
        print(f"Attempting to save full model to {save_dir}/full_model")
        model.save(f"{save_dir}/full_model")
        print("Full model saved successfully")
    except Exception as e:
        print(f"Error saving full model: {e}")
        
        # Fallback 1: Save model weights
        try:
            print("Attempting to save model weights")
            model.save_weights(f"{save_dir}/model_weights")
            print(f"Model weights saved to {save_dir}/model_weights")
            
            # Save model architecture as JSON
            try:
                config = model.get_config()
                with open(f"{save_dir}/model_config.json", 'w') as f:
                    json.dump(config, f)
                print(f"Model architecture saved to {save_dir}/model_config.json")
            except Exception as e:
                print(f"Could not save model architecture: {e}")
                
        except Exception as e:
            print(f"Error saving model weights: {e}")
            
            # Fallback 2: Save trainable weights as NumPy arrays
            try:
                print("Attempting to save individual layer weights as NumPy arrays")
                import numpy as np
                
                weights_dir = f"{save_dir}/layer_weights"
                os.makedirs(weights_dir, exist_ok=True)
                
                for i, w in enumerate(model.trainable_weights):
                    weight_path = f"{weights_dir}/weight_{i}.npy"
                    np.save(weight_path, w.numpy())
                
                # Save weight names/shapes for reference
                weight_info = [
                    {"name": w.name, "shape": w.shape.as_list(), "index": i}
                    for i, w in enumerate(model.trainable_weights)
                ]
                
                with open(f"{weights_dir}/weight_info.json", 'w') as f:
                    json.dump(weight_info, f)
                    
                print(f"Layer weights saved to {weights_dir}")
            except Exception as e:
                print(f"Error saving layer weights: {e}")
                
    return save_dir

def load_model_with_fallbacks(save_dir):
    """Try multiple methods to load a saved model"""
    import os
    import tensorflow as tf
    import numpy as np
    import json
    from architecture.DSTGCN import DSTGCN
    
    # Try loading full model
    if os.path.exists(f"{save_dir}/full_model"):
        try:
            print(f"Attempting to load full model from {save_dir}/full_model")
            model = tf.keras.models.load_model(f"{save_dir}/full_model")
            print("Full model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading full model: {e}")
    
    # Try loading model weights
    if os.path.exists(f"{save_dir}/model_weights.index"):
        try:
            print("Attempting to load model from architecture + weights")
            
            # Load architecture from config
            if os.path.exists(f"{save_dir}/model_config.json"):
                with open(f"{save_dir}/model_config.json", 'r') as f:
                    config = json.load(f)
                    
                # Create model from config
                model = DSTGCN.from_config(config)
                
                # Load weights
                model.load_weights(f"{save_dir}/model_weights")
                print("Model loaded from architecture + weights")
                return model
            else:
                print("Model config not found")
        except Exception as e:
            print(f"Error loading model from architecture + weights: {e}")
    
    # Try loading from checkpoint
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        try:
            print("Attempting to load latest checkpoint")
            # Get latest checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("weights.")]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                
                # Create model
                # You'll need to use the same parameters as original model
                model = DSTGCN((25, 2, 21, 42, 473))  # Example values
                
                # Compile model
                model.compile(
                    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.MeanAbsoluteError())
                
                # Load weights
                model.load_weights(checkpoint_path)
                print(f"Model loaded from checkpoint: {checkpoint_path}")
                return model
        except Exception as e:
            print(f"Error loading from checkpoint: {e}")
    
    print("Failed to load model using any method")
    return None

if __name__ == "__main__":
    start_proc = time.time()
    spatial_data = pd.read_parquet("data/final_data/spatial_data.parquet")
    external_2023 = pd.read_parquet("data/final_data/external_2023.parquet")
    external_2024 = pd.read_parquet("data/final_data/external_2024.parquet")
    temporal_2023 = pd.read_parquet("data/final_data/temporal_2023.parquet")
    temporal_2024 = pd.read_parquet("data/final_data/temporal_2024.parquet")
    graph = pd.read_pickle("data/subway_network.pkl")
    print(f'loaded files in {time.time()-start_proc:4f}s')
    
    # Check for saved models to resume training
    save_dirs = []
    if os.path.exists("model/saved_models"):
        save_dirs = sorted([d for d in os.listdir("model/saved_models") 
                         if os.path.isdir(os.path.join("model/saved_models", d))])
    
    if save_dirs:
        print(f"Found {len(save_dirs)} saved models. Attempting to load the latest one.")
        latest_dir = os.path.join("model/saved_models", save_dirs[-1])
        model = load_model_with_fallbacks(latest_dir)
        
        if model is None:
            print("Could not load saved model. Creating a new one.")
            model_load = time.time()
            model = DSTGCN((len(spatial_data.columns),
                          len(temporal_2023.columns),
                          21, 
                          42,
                          len(spatial_data)))
            print(f'instantiated new model in {time.time() - model_load:4f}s')
    else:
        model_load = time.time()
        model = DSTGCN((len(spatial_data.columns),
                      len(temporal_2023.columns),
                      21, 
                      42,
                      len(spatial_data)))
        print(f'instantiated new model in {time.time() - model_load:4f}s')
    
    adjacency_matrix = build_adjacency_matrix(graph)
    epochs = 10
    batch_size = 128
    
    print(f'starting to train')
    training_start = time.time()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsoluteError())
    
    # Create directories for saved models and checkpoints
    os.makedirs("model/saved_models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train with checkpointing
    train(model=model, epochs=epochs, batch_size=batch_size, 
          data=(spatial_data, temporal_2023, external_2023, adjacency_matrix),
          use_checkpoints=True)
    
    # Save model with multiple fallbacks
    save_dir = save_model_with_fallbacks(model)
    print(f"Model saved to {save_dir}")
    
    # Testing
    average_batch_loss = test(model=model, batch_size=batch_size, 
                             data=(spatial_data, temporal_2024, external_2024, adjacency_matrix))
    print(f"Test loss: {average_batch_loss}")
    
    # Save test results
    try:
        with open(f"{save_dir}/test_results.txt", "w") as f:
            f.write(f"Test loss: {average_batch_loss}\n")
        print(f"Test results saved to {save_dir}/test_results.txt")
    except Exception as e:
        print(f"Error saving test results: {e}")