# train.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import Sen2FireDataset
from augmentations import Compose, Standardize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90
from model import get_model
from loss import get_loss_function
from torch.amp import autocast, GradScaler
from metrics import calculate_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp, loss_name):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False, colour="green")

    for batch in progress_bar:
        inputs, labels = batch['input'].to(device), batch['label'].to(device)
        
        # --- MODIFICATION REMOVED ---
        # The label shape is now correct for BCE_DICE by default

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    print(f"Non-finite loss {loss.item()}; skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print(f"Non-finite loss {loss.item()}; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device, use_amp, loss_name):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False, colour="yellow")
        for batch in progress_bar:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)

            # --- MODIFICATION REMOVED ---
            
            if use_amp:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def estimate_best_threshold(model, dataloader, device, model_name):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['input'].to(device), batch['label'].to(device)
            outputs = model(x)
            
            # --- MODIFICATION REMOVED ---
            # The output is now already single-channel, so no adaptation is needed.
            
            p = torch.sigmoid(outputs).detach().cpu().view(-1)
            y = y.detach().cpu().view(-1)
            all_probs.append(p); all_labels.append(y)
            
    probs = torch.cat(all_probs); labels = torch.cat(all_labels)
    best_t, best_f1 = 0.5, 0.0
    for t in torch.linspace(0.05, 0.95, steps=19):
        preds = (probs > t).float()
        tp = (preds * labels).sum()
        fp = preds.sum() - tp
        fn = labels.sum() - tp
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1

def main(config):
    # --- Setup Unique Run Directory ---
    run_name = config['base_run_name']
    run_path = os.path.join(config['output_dir'], run_name)
    # MODIFIED: Allow overwriting for easier re-runs
    # if os.path.exists(run_path):
    #     raise FileExistsError(f"...")
    os.makedirs(run_path, exist_ok=True)
    print(f"Starting run: {run_name}")
    print(f"Output will be saved to: {run_path}")

    # --- Device ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

    # --- Load Stats (mean/std and optional pos_weight) ---
    stats_path = config['stats_file']
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Statistics file not found at '{stats_path}'.\n"
            f"Please run `preprocessing.py` first to generate it."
        )
    stats = torch.load(stats_path, map_location='cpu')
    mean, std = stats['mean'], stats['std']
    pos_weight = None
    if config.get('use_pos_weight_from_stats', False) and 'pos_weight' in stats:
        pos_weight = torch.tensor(stats['pos_weight'], dtype=torch.float32, device=device)
        print(f"Using pos_weight from stats: {stats['pos_weight']:.4f}")

    print(f"Successfully loaded normalization statistics from {stats_path}")

    # --- Transforms ---
    train_transform = Compose([
        Standardize(mean, std),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5)
    ])
    val_transform = Compose([
        Standardize(mean, std)
    ])

    # --- Data ---
    train_dataset = Sen2FireDataset(config['data_path'], config['train_scenes'], train_transform)
    val_dataset = Sen2FireDataset(config['data_path'], config['val_scenes'], val_transform)

    weights_path = "runs/sampler_weights.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Sampler weights not found at {weights_path}. Please run calculate_weights.py first.")
        
    print(f"Loading pre-calculated sampler weights from {weights_path}")
    weights = torch.load(weights_path)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda')
    )
    print(f"Training data: {len(train_dataset)} samples. Validation data: {len(val_dataset)} samples.")

    # --- Model, Loss, Optimizer, Scheduler ---
    model = get_model(
        model_name=config['model_name'],
        input_channels=config['input_channels'],
        output_channels=1 # MODIFIED: Explicitly ask for 1 output channel
    )
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using nn.DataParallel for {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    loss_name = config.get('loss_name', 'BCE')
    criterion = get_loss_function(
        loss_name=loss_name,
        pos_weight=pos_weight,
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        dice_smooth=config.get('dice_smooth', 1.0)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    # --- AMP ---
    use_amp = bool(config.get('amp', True)) and (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    # --- Train Loop ---
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0
    patience = config.get('patience', 20)

    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp, loss_name)
        val_loss = validate_one_epoch(model, val_loader, criterion, device, use_amp, loss_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state, os.path.join(run_path, 'best_model.pth'))
            print(f"Validation loss improved to {best_val_loss:.4f}. New best model saved!")
            patience_counter = 0
            
            best_t, best_f1 = estimate_best_threshold(model, val_loader, device, config['model_name'])
            with open(os.path.join(run_path, 'best_threshold.txt'), 'w') as f:
                f.write(str(best_t))
            print(f"Saved best threshold {best_t:.2f} (val F1 ~ {best_f1:.3f})")

        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # --- Plot losses ---
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Loss Curve for {run_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_path, 'loss_curve.png'))
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"\nTraining complete. Model and plots saved in {run_path}")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)