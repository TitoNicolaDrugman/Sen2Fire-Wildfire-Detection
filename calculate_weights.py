# calculate_weights.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from dataset import Sen2FireDataset
from torch.utils.data import DataLoader

def calculate_pixel_weight_map(config):
    """
    Calculates a weight map based on the frequency of the positive class at each pixel location.
    """
    print("Calculating per-pixel weight map...")

    # --- Dataset and Loader (no transforms, no shuffle) ---
    train_dataset = Sen2FireDataset(config['data_path'], config['train_scenes'], transform=None)
    loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    num_samples = len(train_dataset)
    fire_counts = torch.zeros(512, 512, dtype=torch.float32)

    progress_bar = tqdm(loader, desc="Counting fire pixels", colour="cyan")
    for batch in progress_bar:
        labels = batch['label'] # (N, 1, H, W)
        fire_counts += labels.sum(dim=0).squeeze(0) # Sum over batch dimension

    # Calculate non-fire counts at each pixel location
    non_fire_counts = num_samples - fire_counts

    # Calculate pos_weight for each pixel: neg_count / pos_count
    epsilon = 1e-7
    pixel_weight_map = non_fire_counts / (fire_counts + epsilon)
    
    # Clip weights to a reasonable range to avoid extreme values for pixels that are never/always fire
    max_weight = config.get('max_pixel_weight', 20.0)
    pixel_weight_map.clamp_(min=1.0, max=max_weight)
    
    # Reshape for loss function: (1, 1, H, W)
    pixel_weight_map = pixel_weight_map.unsqueeze(0).unsqueeze(0)

    # --- Save the map ---
    weights_path = "runs/pixel_weight_map.pt"
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(pixel_weight_map, weights_path)
    
    print(f"\nPixel weight map saved to {weights_path}")
    print(f"Map shape: {pixel_weight_map.shape}")
    print(f"Min weight: {pixel_weight_map.min():.2f}, Max weight: {pixel_weight_map.max():.2f}, Mean weight: {pixel_weight_map.mean():.2f}")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Add a default for max_pixel_weight if not in config
    if 'max_pixel_weight' not in config:
        config['max_pixel_weight'] = 100.0
    calculate_pixel_weight_map(config)