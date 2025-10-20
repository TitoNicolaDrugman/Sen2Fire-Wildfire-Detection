# loss.py
import torch.nn as nn

def get_loss_function():
    """
    Returns BCEWithLogitsLoss, which is numerically stable and suitable for
    binary (fire/no-fire) segmentation tasks.
    """
    return nn.BCEWithLogitsLoss()