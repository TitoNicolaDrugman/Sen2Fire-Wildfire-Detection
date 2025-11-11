# preprocessing.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import Sen2FireDataset

def calculate_and_save_stats(config):
    """
    Calculates and saves the channel-wise mean and standard deviation 
    for the training dataset to avoid data leakage.
    """
    print("Calculating dataset statistics (mean and std)...")
    
    # --- Dataset and DataLoader ---
    # We use the training scenes and apply no transformations for this calculation
    train_dataset = Sen2FireDataset(config['data_path'], config['train_scenes'], transform=None)
    # Use a larger batch size to speed up calculation
    loader = DataLoader(train_dataset, batch_size=config['batch_size']*2, shuffle=False, num_workers=config['num_workers'])

    # --- Calculation ---
    # We will calculate a running mean and standard deviation
    # for all 13 channels (12 Sentinel-2 + 1 Sentinel-5P aerosol)
    num_channels = config['input_channels']
    sum_ = torch.zeros(num_channels)
    sum_sq = torch.zeros(num_channels)
    num_pixels = 0
    pixels_per_sample = 512 * 512

    progress_bar = tqdm(loader, desc="Calculating Stats", colour="cyan")
    for batch in progress_bar:
        # 'input' is already the concatenated (image, aerosol) tensor
        inputs = batch['input']  # Shape: (N, C, H, W)
        
        # Sum over all dimensions except the channel dimension (C)
        sum_ += torch.sum(inputs, dim=[0, 2, 3])
        sum_sq += torch.sum(inputs ** 2, dim=[0, 2, 3])
        num_pixels += inputs.shape[0] * pixels_per_sample

    # --- Finalize Mean and Std ---
    mean = sum_ / num_pixels
    std = torch.sqrt((sum_sq / num_pixels) - (mean ** 2))

    # Reshape for broadcasting during normalization: (C) -> (C, 1, 1)
    mean = mean.view(num_channels, 1, 1)
    std = std.view(num_channels, 1, 1)
    
    # --- Save to File ---
    stats_file_path = config['stats_file']
    os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
    torch.save({'mean': mean, 'std': std}, stats_file_path)

    print(f"\nStatistics saved successfully to {stats_file_path}")
    print(f"  - Mean shape: {mean.shape}")
    print(f"  - Std shape: {std.shape}")
    
    return {'mean': mean, 'std': std}

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    calculate_and_save_stats(config)