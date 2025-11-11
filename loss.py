# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        outputs = torch.sigmoid(outputs) # Apply sigmoid to get probabilities
        
        # Flatten label and prediction tensors
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        
        intersection = (outputs * labels).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + labels.sum() + self.smooth)
        
        return 1 - dice

def get_loss_function(loss_name='BCE'):
    """Returns a loss function based on a name."""
    if loss_name.upper() == 'BCE':
        print("Using BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    elif loss_name.upper() == 'DICE':
        print("Using DiceLoss")
        return DiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")