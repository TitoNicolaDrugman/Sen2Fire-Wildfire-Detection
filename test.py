
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


# ---------------- Utilities ----------------
def select_device(cfg_device: str):
    if cfg_device and "cuda" in cfg_device and torch.cuda.is_available():
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state_dict_flexible(model, path):
    """
    Loads weights saved possibly from DataParallel.
    """
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        # strip 'module.' if model not wrapped
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


# ---------------- TTA (returns probabilities) ----------------
def predict_with_tta_probs(model, x):
    """
    Returns averaged probabilities from TTA transforms.
    model(x) -> logits (B,1,H,W). We sigmoid then average probs.
    """
    device = next(model.parameters()).device
    x = x.to(device)

    outs = []

    def infer(inp):
        out = model(inp)           # logits
        prob = torch.sigmoid(out) # probs
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
        outs.append(torch.rot90(pr, -k, dims=[2, 3]))

    probs_avg = torch.stack(outs, dim=0).mean(dim=0)  # (B,1,H,W)
    return probs_avg


# ---------------- Local metric computation (binary) ----------------
def compute_binary_metrics(preds_np, labels_np):
    preds = preds_np.flatten().astype(np.uint8)
    labels = labels_np.flatten().astype(np.uint8)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1, "iou": iou}


def find_best_threshold(all_probs_np, all_labels_np, thr_values=None):
    if thr_values is None:
        thr_values = np.linspace(0.01, 0.99, 99)

    probs = np.concatenate(all_probs_np, axis=0).ravel()
    labels = np.concatenate(all_labels_np, axis=0).ravel().astype(np.uint8)

    best_thr = 0.5
    best_f1 = 0.0

    for thr in thr_values:
        preds = (probs >= thr).astype(np.uint8)
        m = compute_binary_metrics(preds, labels)
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_thr = thr

    return best_thr, best_f1


# ---------------- Main testing logic ----------------
def test_model(config, run_name):
    device = select_device(config.get("device", ""))
    print(f"Using device: {device} for testing")

    # --- Paths ---
    model_path = os.path.join(config["output_dir"], run_name, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading model for evaluation: {model_path}")

    stats_path = config["stats_file"]
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at '{stats_path}'. Run preprocessing.py first.")
    stats = torch.load(stats_path, map_location="cpu")
    mean, std = stats["mean"], stats["std"]
    print("Loaded normalization statistics for testing.")

    # --- Strategy ---
    strategy = config.get("input_strategy", "vanilla")

    # --- Dataset/DataLoader ---
    test_transform = Compose([Standardize(mean, std)])
    test_dataset = Sen2FireDataset(
        config["data_path"],
        config["test_scenes"],
        transform=test_transform,
        input_strategy=strategy
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda")
    )
    print(f"Test scenes: {config['test_scenes']}")
    print(f"Test samples: {len(test_dataset)}")

    # Infer channels from dataset
    in_channels = test_dataset[0]["input"].shape[0]
    print(f"Detected {in_channels} input channels for strategy '{strategy}'")

    # --- Model ---
    model = get_model(
        model_name=config.get("model_name", "unet"),
        input_channels=in_channels,
        n_classes=config.get("n_classes", 1)
    )
    model = load_state_dict_flexible(model, model_path)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    # --- TTA ---
    use_tta = config.get("use_tta", True)
    print(f"\nTest-Time Augmentation (TTA) is {'ENABLED' if use_tta else 'DISABLED'}.")

    all_probs_np = []
    all_labels_np = []

    with torch.no_grad():
        desc = f"Testing {run_name}" + (" with TTA" if use_tta else " without TTA")
        progress_bar = tqdm(test_loader, desc=desc, colour="blue")

        for batch in progress_bar:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device).float()

            if use_tta:
                probs = predict_with_tta_probs(model, inputs)  # (B,1,H,W)
            else:
                logits = model(inputs)
                probs = torch.sigmoid(logits)

            # squeeze channel dim for numpy concat
            probs = probs.squeeze(1)   # (B,H,W)
            labels = labels.squeeze(1) # (B,H,W)

            all_probs_np.append(probs.detach().cpu().numpy())
            all_labels_np.append(labels.detach().cpu().numpy())

    # --- Threshold selection ---
    eval_threshold = float(config.get("eval_threshold", 0.5))
    do_find_best_thr = bool(config.get("find_best_threshold", True))

    if do_find_best_thr:
        print("Searching best threshold on test set (set find_best_threshold=False to disable)...")
        best_thr, best_f1 = find_best_threshold(all_probs_np, all_labels_np)
        print(f"Best threshold found: {best_thr:.3f} (F1={best_f1:.4f})")
        chosen_thr = best_thr
    else:
        chosen_thr = eval_threshold
        print(f"Using provided eval_threshold = {chosen_thr}")

    # --- Final metrics ---
    probs_concat = np.concatenate(all_probs_np, axis=0)
    labels_concat = np.concatenate(all_labels_np, axis=0)

    preds_concat = (probs_concat >= chosen_thr).astype(np.uint8)

    final_metrics = compute_binary_metrics(preds_concat, labels_concat)

    print("\n--- Test Results (computed here) ---")
    print(f"Precision   : {final_metrics['precision']:.4f}")
    print(f"Recall      : {final_metrics['recall']:.4f}")
    print(f"F1 score    : {final_metrics['f1_score']:.4f}")
    print(f"IoU         : {final_metrics['iou']:.4f}")

    # --- Save results ---
    results_file = config["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "a") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Results for run: {run_name}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"TTA Enabled: {use_tta}\n")
        f.write(f"Model: {config.get('model_name','unet')}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Chosen threshold: {chosen_thr:.4f}\n")
        for k, v in final_metrics.items():
            f.write(f"{k.replace('_',' ').capitalize():<12}: {v:.4f}\n")
        f.write("=" * 50 + "\n\n")

    print(f"\nResults appended to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Sen2Fire model.")
    parser.add_argument("run_name", type=str, help="Run folder name inside /runs")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    test_model(config, args.run_name)
