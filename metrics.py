# metrics.py
import torch

def calculate_metrics(outputs, labels, threshold=0.5):
    """
    Calculates precision, recall, F1-score, and IoU for a binary segmentation task.
    """
    preds = torch.sigmoid(outputs) > threshold
    preds = preds.float()
    labels = labels.float()

    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    tp = (preds_flat * labels_flat).sum()
    fp = preds_flat.sum() - tp
    fn = labels_flat.sum() - tp
    
    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1.item(),
        'iou': iou.item()
    }