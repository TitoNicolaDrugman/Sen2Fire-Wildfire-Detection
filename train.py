# train.py
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from dataset import Sen2FireDataset
from augmentations import NoAugmentation
from model import SimpleMLP
from loss import get_loss_function

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False, colour="green")
    for batch in progress_bar:
        inputs, labels = batch['input'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False, colour="yellow")
        for batch in progress_bar:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def main(config):
    # --- Setup Unique Run Directory ---
    run_name = config['base_run_name']
    run_path = os.path.join(config['output_dir'], run_name)

    # --- Prevent Overwriting ---
    if os.path.exists(run_path):
        raise FileExistsError(
            f"Run directory '{run_path}' already exists.\n"
            f"Please change the 'base_run_name' in 'config.yaml' to a unique name to avoid overwriting results."
        )
    
    os.makedirs(run_path)
    print(f"Starting run: {run_name}")
    print(f"Output will be saved to: {run_path}")

    # --- Setup Device and Log GPU Info ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # --- Data Loading ---
    transform = NoAugmentation()
    train_dataset = Sen2FireDataset(config['data_path'], config['train_scenes'], transform)
    val_dataset = Sen2FireDataset(config['data_path'], config['val_scenes'], transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # --- Model, Loss, and Optimizer ---
    model = SimpleMLP(input_channels=config['input_channels'])
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using DataParallel for multi-GPU training.")
        model = nn.DataParallel(model)

    criterion = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state, os.path.join(run_path, 'best_model.pth'))
            print("New best model saved!")

    # --- Plotting and Saving ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curve for {run_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_path, 'loss_curve.png'))
    print(f"\nTraining complete. Model and plots saved in {run_path}")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)