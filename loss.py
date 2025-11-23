# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N,1,H,W), targets: (N,1,H,W)
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # binary focal loss with logits
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal = (self.alpha * (1 - pt) ** self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1); targets = targets.view(-1)
        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1.0 - tversky

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma
    def forward(self, logits, targets):
        t = self.tversky(logits, targets)
        return torch.pow(t, self.gamma)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, dice_smooth=1.0, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight 
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight
        )
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice






def get_loss_function(loss_name='BCE', pos_weight=None, focal_alpha=0.25, focal_gamma=2.0, dice_smooth=1.0):
    """
    Returns a loss function based on a name.
    - BCE: BCEWithLogitsLoss, can accept pos_weight
    - DICE: DiceLoss
    - BCE_DICE: Combined BCE + Dice
    - FOCAL: Focal loss
    """
    name = loss_name.upper()
    if name == 'BCE':
        print("Using BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'DICE':
        print("Using DiceLoss")
        return DiceLoss(smooth=dice_smooth)
    elif name == 'BCE_DICE':
        print("Using BCEDiceLoss (0.5 * BCE + 0.5 * Dice)")
        return BCEDiceLoss(pos_weight=pos_weight, dice_smooth=dice_smooth)
    elif name == 'FOCAL':
        print("Using FocalLoss")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif name == 'FOCAL_TVERSKY':
        print("Using FocalTverskyLoss")
        return FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=focal_gamma, smooth=dice_smooth)
    elif name == 'BCE_FT':
        print("Using BCE + FocalTverskyLoss (0.3 * BCE + 0.7 * FT)")
        class _BCEFT(nn.Module):
            def __init__(self, pos_weight, gamma, smooth):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                self.ft  = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=gamma, smooth=smooth)
            def forward(self, logits, targets):
                return 0.3 * self.bce(logits, targets) + 0.7 * self.ft(logits, targets)
        return _BCEFT(pos_weight, focal_gamma, dice_smooth)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
