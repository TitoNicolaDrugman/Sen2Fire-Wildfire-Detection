# calculate_weights.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from dataset import Sen2FireDataset

print("Calculating sampler weights...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train_dataset = Sen2FireDataset(config['data_path'], config['train_scenes'])

pos_frac = 0.025023 # You can get this from your preprocessing output
pos_boost = max(2.0, min(20.0, 0.7 / max(pos_frac, 1e-6)))

weights = []
for path in tqdm(train_dataset.file_paths, desc="Reading labels"):
    with np.load(path) as d:
        has_pos = d['label'].sum() > 0
    weights.append(pos_boost if has_pos else 1.0)

weights_tensor = torch.DoubleTensor(weights)
weights_path = "runs/sampler_weights.pt"
torch.save(weights_tensor, weights_path)
print(f"Sampler weights saved to {weights_path}")