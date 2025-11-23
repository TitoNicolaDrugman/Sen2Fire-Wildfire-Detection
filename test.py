# test.py

import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from dataset import Sen2FireDataset
from augmentations import Compose, Standardize
from model import get_model
from metrics import calculate_metrics

def predict_with_tta(model, x):
    """
    Performs Test-Time Augmentation by making predictions on the original,
    flipped, and rotated versions of the input, then averaging the results.
    x: (N, C, H, W) input tensor
    """
    outs = []
    # 1. Identity
    outs.append(model(x))
    # 2. Horizontal flip
    xh = torch.flip(x, [3])
    outs.append(torch.flip(model(xh), [3]))
    # 3. Vertical flip
    xv = torch.flip(x, [2])
    outs.append(torch.flip(model(xv), [2]))
    # 4, 5, 6. Rotations (90, 180, 270 degrees)
    for k in [1, 2, 3]:
        xr = torch.rot90(x, k, dims=[2, 3])
        yr = model(xr)
        # Rotate back to original orientation
        outs.append(torch.rot90(yr, -k, dims=[2, 3]))

    # Stack all predictions and average them
    return torch.stack(outs, dim=0).mean(dim=0)


def test_model(config, test_run_name):
    """Evaluates a trained model on the test dataset."""
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for testing")

    # --- Paths ---
    model_path = os.path.join(config['output_dir'], test_run_name, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading model for evaluation: {model_path}")

    # --- 1. Load normalization stats ---
    stats_path = config['stats_file']
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at '{stats_path}'. Run preprocessing.py.")
    stats = torch.load(stats_path, map_location='cpu')
    mean, std = stats['mean'], stats['std']
    print("Loaded normalization statistics for testing.")

    # --- 2. Test transforms ---
    test_transform = Compose([Standardize(mean, std)])

    # --- 3. Dataset/DataLoader ---
    test_dataset = Sen2FireDataset(config['data_path'], config['test_scenes'], test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda')
    )
    print(f"Test scenes: {config['test_scenes']}")
    print(f"Test samples: {len(test_dataset)}")

    # --- 4. Load Model ---
    model = get_model(
        model_name=config['model_name'],
        input_channels=config['input_channels']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 5. Evaluation ---
    # Get TTA setting from config, defaulting to True if not specified
    use_tta = config.get('use_tta', True)
    print(f"\nTest-Time Augmentation (TTA) is {'ENABLED' if use_tta else 'DISABLED'}.")

    all_metrics = {'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    th_path = os.path.join(config['output_dir'], test_run_name, 'best_threshold.txt')
    if os.path.exists(th_path):
        with open(th_path, 'r') as f:
            eval_threshold = float(f.read().strip())
    else:
        eval_threshold = float(config.get('eval_threshold', 0.5))

    with torch.no_grad():
        desc = f"Testing {test_run_name}" + (" with TTA" if use_tta else " without TTA")
        progress_bar = tqdm(test_loader, desc=desc, colour="blue")
        for batch in progress_bar:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            # Conditionally perform prediction with or without TTA
            if use_tta:
                outputs = predict_with_tta(model, inputs)
            else:
                outputs = model(inputs)

            batch_metrics = calculate_metrics(outputs, labels, threshold=eval_threshold)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])

    # --- 6. Aggregate and save results ---
    avg_metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
    print("\n--- Test Results ---")
    for key, value in avg_metrics.items():
        print(f"{key.replace('_', ' ').capitalize():<12}: {value:.4f}")

    results_file = config['results_file']
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'a') as f:
        f.write("="*50 + "\n")
        f.write(f"Results for run: {test_run_name}\n")
        f.write(f"TTA Enabled: {use_tta}\n")  # Log TTA status to file
        f.write(f"Model Architecture: {config['model_name']}\n")
        f.write(f"Model Path: {model_path}\n")
        for key, value in avg_metrics.items():
            f.write(f"{key.replace('_', ' ').capitalize():<12}: {value:.4f}\n")
        f.write("="*50 + "\n\n")
    print(f"\nResults appended to {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained Sen2Fire model.")
    parser.add_argument(
        'run_name',
        type=str,
        help=("The unique name of the run folder to test (e.g., 'ConvTransUNet_imbalance_run1'), "
              "which matches 'base_run_name' used during training.")
    )
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    test_model(config, args.run_name)