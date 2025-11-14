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
    Very simple encoder–decoder CNN for segmentation.
    Input:  (B, C, H, W)
    Output: (B, num_classes, H, W) logits
    """
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)   # H/2

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)   # H/4

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)   # H/8

        # Decoder (upsampling back to original size)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)
        x = self.pool3(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        # NO sigmoid here – let the loss handle logits
        return x