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
    for the training dataset.
    """
    print("Calculating dataset statistics (mean and std)...")
    
    # --- Dataset and DataLoader ---
    # We don't need any transforms for this calculation
    train_dataset = Sen2FireDataset(config['data_path'], config['train_scenes'], transform=None)
    # Use a larger batch size to speed up calculation, num_workers can also be increased
    loader = DataLoader(train_dataset, batch_size=config['batch_size']*2, shuffle=False, num_workers=config['num_workers'])

    # --- Calculation ---
    # These will be used to calculate a running mean and std
    # See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    num_channels = config['input_channels']
    sum_ = torch.zeros(num_channels)
    sum_sq = torch.zeros(num_channels)
    num_pixels = 0

    # The dataset size is 512x512
    pixels_per_sample = 512 * 512

    progress_bar = tqdm(loader, desc="Calculating Stats", colour="cyan")
    for batch in progress_bar:
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

    print(f"Statistics saved successfully to {stats_file_path}")
    print(f"  - Mean shape: {mean.shape}")
    print(f"  - Std shape: {std.shape}")
    
    return {'mean': mean, 'std': std}


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    calculate_and_save_stats(config)