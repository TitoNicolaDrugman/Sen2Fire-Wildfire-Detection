# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# this is a stupid model just for checking if all the code is running correctyly
class SimpleMLP(nn.Module):
    """
    A simple pixel-wise MLP model for wildfire detection.
    It treats each pixel independently.
    """
    def __init__(self, input_channels, output_channels=1):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 16),
            nn.ReLU(),
            nn.Linear(16, output_channels)
        )

    def forward(self, x):
        # x shape: (N, C, H, W) -> (N, 13, 512, 512)
        x = x.permute(0, 2, 3, 1)  # New shape: (N, H, W, C)
        x = self.mlp(x)            # Output shape: (N, H, W, 1)
        x = x.permute(0, 3, 1, 2)  # New shape: (N, 1, H, W)
        return x


class SimpleCNN(nn.Module):
    """
    Small convolutional model suitable for quick experiments.
    If n_classes == 1 -> outputs a single-channel map (segmentation / binary mask logits).
    If n_classes > 1 -> outputs class logits (for classification use global pooling).
    """
    def __init__(self, input_channels=3, n_classes=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )

        # decoder / head
        # use conv head for segmentation-style output (keeps spatial dims)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # /4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # /2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # original size
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, kernel_size=1)
        )

        # simple classification head (global pooling -> linear)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes)
        )

        self.n_classes = n_classes

    def forward(self, x):
        x = self.encoder(x)  # (B, 128, H/8, W/8)
        if self.n_classes == 1:
            # segmentation/binary mask logits
            return self.seg_head(x)
        else:
            # classification logits
            return self.class_head(x)

def get_model(model_name: str = "simple_cnn", input_channels: int = 3, n_classes: int = 1):
    if model_name.lower() in ("simple_cnn", "cnn", "conv"):
        return SimpleCNN(input_channels=input_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")