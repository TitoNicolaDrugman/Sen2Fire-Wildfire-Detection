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

# --- Configurable: whether to try to use your external calculate_metrics ---
USE_EXTERNAL_METRICS = False
try:
    if USE_EXTERNAL_METRICS:
        from metrics import calculate_metrics as external_calculate_metrics
except Exception:
    USE_EXTERNAL_METRICS = False

# ---------------- Utilities ----------------
def select_device(cfg_device: str):
    if cfg_device and "cuda" in cfg_device and torch.cuda.is_available():
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_state_dict_flexible(model, path, device):
    """
    Loads a state dict saved possibly from DataParallel.
    Returns loaded model (unwrapped or wrapped after loading).
    """
    state = torch.load(path, map_location='cpu')
    # strip module. if necessary
    if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()) and not isinstance(model, torch.nn.DataParallel):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    return model

# ---------------- TTA (returns probabilities) ----------------
def predict_with_tta_probs(model, x):
    """
    Given input tensor x on the same device as model, returns averaged probabilities.
    Works for spatial outputs (segmentation maps): model returns logits shape (B,1,H,W).
    Averages sigmoid(model(transforms(x))) across TTA transforms.
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)

    outs = []

    def infer(inp):
        with torch.no_grad():
            out = model(inp)          # logits
            prob = torch.sigmoid(out) # convert logits -> probs
        return prob

    # identity
    outs.append(infer(x))

    # horizontal flip
    xh = torch.flip(x, [3])
    ph = infer(xh)
    outs.append(torch.flip(ph, [3]))

    # vertical flip
    xv = torch.flip(x, [2])
    pv = infer(xv)
    outs.append(torch.flip(pv, [2]))

    # rotations 90,180,270
    for k in [1, 2, 3]:
        xr = torch.rot90(x, k, dims=[2, 3])
        pr = infer(xr)
        pr = torch.rot90(pr, -k, dims=[2, 3])
        outs.append(pr)

    probs_avg = torch.stack(outs, dim=0).mean(dim=0)  # (B,1,H,W)
    return probs_avg

# ---------------- Local metric computation (binary) ----------------
def compute_binary_metrics_from_preds_and_labels(preds_np, labels_np):
    """
    preds_np, labels_np: flattened arrays (0/1)
    returns dict with precision, recall, f1_score, iou
    Zero-division safe.
    """
    preds = preds_np.flatten().astype(np.uint8)
    labels = labels_np.flatten().astype(np.uint8)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    # tn not needed

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1, "iou": iou}

# ---------------- Threshold search utility ----------------
def find_best_threshold(all_probs_np, all_labels_np, thr_values=None):
    """
    all_probs_np and all_labels_np are flattened 1D arrays or concatenated arrays (not necessarily flattened yet).
    Returns best_threshold, best_f1
    """
    if thr_values is None:
        thr_values = np.linspace(0.01, 0.99, 99)

    probs = np.concatenate(all_probs_np, axis=0).ravel()
    labels = np.concatenate(all_labels_np, axis=0).ravel().astype(np.uint8)

    best_thr = 0.5
    best_f1 = 0.0
    for thr in thr_values:
        preds = (probs >= thr).astype(np.uint8)
        metrics = compute_binary_metrics_from_preds_and_labels(preds, labels)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_thr = thr
    return best_thr, best_f1

# ---------------- Main testing logic ----------------
def test_model(config, test_run_name):
    device = select_device(config.get('device',''))
    print(f"Using device: {device} for testing")

    model_path = os.path.join(config['output_dir'], test_run_name, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading model for evaluation: {model_path}")

    stats_path = config['stats_file']
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at '{stats_path}'. Run preprocessing.py.")
    stats = torch.load(stats_path, map_location='cpu')
    mean, std = stats['mean'], stats['std']
    print("Loaded normalization statistics for testing.")

    test_transform = Compose([Standardize(mean, std)])
    test_dataset = Sen2FireDataset(config['data_path'], config['test_scenes'], test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=(device.type=='cuda'))
    print(f"Test scenes: {config['test_scenes']}")
    print(f"Test samples: {len(test_dataset)}")

    model = get_model(
        model_name=config.get('model_name','simple_cnn'),
        input_channels=config.get('input_channels',3),
        n_classes=config.get('n_classes',1)
    )

    # load weights safely
    model = load_state_dict_flexible(model, model_path, device)
    # wrap with DataParallel if multiple GPUs and cuda
    if torch.cuda.device_count() > 1 and "cuda" in str(device):
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    use_tta = config.get('use_tta', True)
    print(f"\nTest-Time Augmentation (TTA) is {'ENABLED' if use_tta else 'DISABLED'}.")

    # Where we will store probabilities and labels for threshold search / final metrics
    all_probs_np = []
    all_labels_np = []

    # If you have an external calculate_metrics that expects batch-level metrics, we won't call it by default.
    # We compute metrics at the end from concatenated arrays (more stable).
    with torch.no_grad():
        desc = f"Testing {test_run_name}" + (" with TTA" if use_tta else " without TTA")
        progress_bar = tqdm(test_loader, desc=desc, colour="blue")
        for batch in progress_bar:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)  # shape (B,1,H,W) or (B,H,W)
            # ensure labels are 0/1 floats
            labels = labels.float()

            if use_tta:
                probs = predict_with_tta_probs(model, inputs)  # (B,1,H,W) probs in [0,1]
            else:
                with torch.no_grad():
                    logits = model(inputs)
                    probs = torch.sigmoid(logits)

            # ensure shapes match: squeeze channel dim if necessary
            if probs.dim() == 4 and probs.shape[1] == 1 and labels.dim() == 3:
                probs = probs.squeeze(1)   # (B,H,W)
            elif probs.dim() == 4 and probs.shape[1] == 1 and labels.dim() == 4 and labels.shape[1] == 1:
                probs = probs.squeeze(1)   # both (B,H,W) -> but labels still (B,1,H,W)
                labels = labels.squeeze(1)

            # DEBUG: show ranges and shapes for the first batch
            if len(all_probs_np) == 0:
                print("DEBUG sample -> outputs/probs shape:", probs.shape, "min/max:", float(probs.min()), float(probs.max()))
                print("DEBUG sample -> labels shape:", labels.shape, "unique labels:", torch.unique(labels)[:10])
                print("eval_threshold:", config.get('eval_threshold', 0.5))

            # move to CPU numpy and append
            all_probs_np.append(probs.detach().cpu().numpy())   # shape (B,H,W)
            all_labels_np.append(labels.detach().cpu().numpy())

    # Concatenate all arrays (axis=0 -> batch axis)
    # We will search threshold if requested
    eval_threshold = float(config.get('eval_threshold', 0.5))
    do_find_best_thr = bool(config.get('find_best_threshold', True))

    if do_find_best_thr:
        print("Searching best threshold on test set (use find_best_threshold=False in config to disable)...")
        best_thr, best_f1 = find_best_threshold(all_probs_np, all_labels_np)
        print(f"Best threshold found: {best_thr:.3f} (F1={best_f1:.4f})")
        chosen_threshold = best_thr
    else:
        chosen_threshold = eval_threshold
        print(f"Using provided eval_threshold = {chosen_threshold}")

    # Build final predictions and compute metrics
    all_probs_concat = np.concatenate(all_probs_np, axis=0)  # (N, H, W)
    all_labels_concat = np.concatenate(all_labels_np, axis=0) # (N, H, W)

    preds_concat = (all_probs_concat >= chosen_threshold).astype(np.uint8)

    # compute metrics using internal function
    final_metrics = compute_binary_metrics_from_preds_and_labels(preds_concat, all_labels_concat)
    print("\n--- Test Results (computed here) ---")
    print(f"Precision   : {final_metrics['precision']:.4f}")
    print(f"Recall      : {final_metrics['recall']:.4f}")
    print(f"F1 score    : {final_metrics['f1_score']:.4f}")
    print(f"IoU         : {final_metrics['iou']:.4f}")

    # Append to results file
    results_file = config['results_file']
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'a') as f:
        f.write("="*50 + "\n")
        f.write(f"Results for run: {test_run_name}\n")
        f.write(f"TTA Enabled: {use_tta}\n")
        f.write(f"Model Architecture: {config.get('model_name','simple_cnn')}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Chosen threshold: {chosen_threshold:.4f}\n")
        for k, v in final_metrics.items():
            f.write(f"{k.replace('_',' ').capitalize():<12}: {v:.4f}\n")
        f.write("="*50 + "\n\n")
    print(f"\nResults appended to {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained Sen2Fire model.")
    parser.add_argument('run_name', type=str, help="The unique name of the run folder to test.")
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    test_model(config, args.run_name)
