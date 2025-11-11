# test.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse 

# --- IMPORT THE NECESSARY AUGMENTATION CLASSES ---
from dataset import Sen2FireDataset
from augmentations import Compose, Standardize 
from model import SimpleMLP
from metrics import calculate_metrics

def test_model(config, test_run_name):
    """Evaluates a trained model on the test dataset."""
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for testing")

    model_path = os.path.join(config['output_dir'], test_run_name, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading model for evaluation: {model_path}")

    # --- 1. LOAD NORMALIZATION STATS ---
    stats_path = config['stats_file']
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at '{stats_path}'. Run preprocessing.py.")
    stats = torch.load(stats_path)
    mean, std = stats['mean'], stats['std']
    print("Loaded normalization statistics for testing.")

    # --- 2. CREATE THE TEST TRANSFORMATION PIPELINE ---
    # The test set is ONLY normalized. No random augmentations.
    test_transform = Compose([
        Standardize(mean, std)
    ])

    # --- 3. APPLY TRANSFORMS TO THE TEST DATASET ---
    test_dataset = Sen2FireDataset(config['data_path'], config['test_scenes'], test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # --- Load Model ---
    model = SimpleMLP(input_channels=config['input_channels'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Evaluation Loop (this part was already correct) ---
    all_metrics = {'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Testing {test_run_name}", colour="blue")
        for batch in progress_bar:
            # Unpack the dictionary from the dataloader
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            batch_metrics = calculate_metrics(outputs, labels)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])

    # ... (rest of the script is fine) ...