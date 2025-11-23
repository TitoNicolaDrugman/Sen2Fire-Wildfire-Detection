# preprocessing.py

import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Sen2FireDataset

def calculate_and_save_stats(config):
    """
    Calculates and saves channel-wise mean/std and class-imbalance stats (pos_weight)
    on the training split only to avoid leakage.
    """
    print("Calculating dataset statistics (mean, std) and class balance...")

    # --- Dataset and Loader (no transforms) ---
    train_dataset = Sen2FireDataset(config['data_path'], config['train_scenes'], transform=None)
    loader = DataLoader(train_dataset, batch_size=config['batch_size']*2, shuffle=False, num_workers=config['num_workers'])

    # --- Running moments for (C,H,W) input ---
    num_channels = config['input_channels']
    sum_ = torch.zeros(num_channels, dtype=torch.float64)
    sum_sq = torch.zeros(num_channels, dtype=torch.float64)
    num_pixels = 0

    # --- Pixel class counts for labels ---
    pos_count = 0
    neg_count = 0
    pixels_per_sample = 512 * 512  # fixed per README/dataset format

    progress_bar = tqdm(loader, desc="Calculating Stats", colour="cyan")
    for batch in progress_bar:
        x = batch['input'].double()   # (N,C,H,W)
        y = batch['label'].double()   # (N,1,H,W)

        sum_ += torch.sum(x, dim=[0, 2, 3])
        sum_sq += torch.sum(x ** 2, dim=[0, 2, 3])
        n = x.shape[0] * pixels_per_sample
        num_pixels += n

        # labels are binary masks in {0,1}
        pos_count += y.sum().item()
        neg_count += n - y.sum().item()

    mean = (sum_ / num_pixels).float().view(num_channels, 1, 1)
    std = torch.sqrt((sum_sq / num_pixels) - (sum_ / num_pixels) ** 2).float().view(num_channels, 1, 1)

    # guard against zero std
    std = torch.clamp(std, min=1e-6)

    # pos_weight = neg/pos for BCEWithLogitsLoss
    eps = 1e-6
    pos_weight = float(neg_count / max(pos_count, eps))
    pos_fraction = float(pos_count / max(pos_count + neg_count, eps))

    stats_file_path = config['stats_file']
    os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
    torch.save(
        {
            'mean': mean,
            'std': std,
            'pos_weight': pos_weight,
            'pos_fraction': pos_fraction,
            'counts': {'pos': pos_count, 'neg': neg_count}
        },
        stats_file_path
    )

    print(f"\nStatistics saved to {stats_file_path}")
    print(f" - Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f" - pos_weight: {pos_weight:.4f}, pos_fraction: {pos_fraction:.6f}")

    return {'mean': mean, 'std': std, 'pos_weight': pos_weight, 'pos_fraction': pos_fraction}

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    calculate_and_save_stats(config)
