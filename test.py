# test.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse 

from dataset import Sen2FireDataset
from model import SimpleMLP
from metrics import calculate_metrics

def test_model(config, test_run_name): # <-- Accept the run name as an argument
    """Evaluates a trained model on the test dataset."""
    # --- Setup Device and Log GPU Info ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for testing")
    if device.type == 'cuda':
        print(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # --- Construct Model Path from the provided run name ---
    model_path = os.path.join(config['output_dir'], test_run_name, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}\n"
                              f"Please ensure the run name '{test_run_name}' exists and contains 'best_model.pth'.")
    print(f"Loading model for evaluation: {model_path}")

    # --- Data Loading ---
    test_dataset = Sen2FireDataset(config['data_path'], config['test_scenes'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # --- Load Model ---
    model = SimpleMLP(input_channels=config['input_channels'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Evaluation Loop ---
    all_metrics = {'precision': [], 'recall': [], 'f1_score': [], 'iou': []}
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Testing {test_run_name}", colour="blue")
        for batch in progress_bar:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            batch_metrics = calculate_metrics(outputs, labels)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])

    # --- Aggregate and Save Results ---
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    print("\n--- Test Results ---")
    for key, value in avg_metrics.items():
        print(f"{key.replace('_', ' ').capitalize():<12}: {value:.4f}")

    results_file = config['results_file']
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'a') as f:
        f.write("="*50 + "\n")
        f.write(f"Results for run: {test_run_name}\n")
        f.write(f"Model Path: {model_path}\n")
        for key, value in avg_metrics.items():
            f.write(f"{key.replace('_', ' ').capitalize():<12}: {value:.4f}\n")
        f.write("="*50 + "\n\n")

    print(f"\nResults appended to {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained Sen2Fire model.")
    parser.add_argument('run_name', type=str, 
                        help="The unique name of the run folder to test (e.g., 'SimpleMLP_baseline_20251020_194736')")
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    test_model(config, args.run_name)