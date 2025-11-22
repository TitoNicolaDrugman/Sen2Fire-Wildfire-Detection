
# preprocessing.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import Sen2FireDataset

def calculate_and_save_stats(config):
    print("Calculating dataset statistics (mean and std)...")

    # build dataset using the same strategy you train with
    train_dataset = Sen2FireDataset(
        config['data_path'],
        config['train_scenes'],
        transform=None,
        input_strategy=config.get('input_strategy', 'vanilla')
    )

    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=config['num_workers']
    )

    # infer channels from real sample
    sample = train_dataset[0]
    num_channels = sample['input'].shape[0]
    print(f"Detected {num_channels} channels for stats calculation "
          f"(strategy='{config.get('input_strategy','vanilla')}')")

    sum_ = torch.zeros(num_channels)
    sum_sq = torch.zeros(num_channels)
    num_pixels = 0
    pixels_per_sample = 512 * 512

    progress_bar = tqdm(loader, desc="Calculating Stats", colour="cyan")
    for batch in progress_bar:
        inputs = batch['input']  # (N, C, H, W)

        sum_ += torch.sum(inputs, dim=[0, 2, 3])
        sum_sq += torch.sum(inputs ** 2, dim=[0, 2, 3])
        num_pixels += inputs.shape[0] * pixels_per_sample

    mean = sum_ / num_pixels
    std = torch.sqrt((sum_sq / num_pixels) - (mean ** 2))

    mean = mean.view(num_channels, 1, 1)
    std = std.view(num_channels, 1, 1)

    stats_file_path = config['stats_file']
    os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
    torch.save({'mean': mean, 'std': std}, stats_file_path)

    print(f"\nStatistics saved successfully to {stats_file_path}")
    print(f"  - Mean shape: {mean.shape}")
    print(f"  - Std shape: {std.shape}")

    return {'mean': mean, 'std': std}


if __name__ == "__main__":
    # load config and run
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    calculate_and_save_stats(config)
