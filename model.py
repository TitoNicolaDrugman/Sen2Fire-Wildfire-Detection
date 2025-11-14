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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Standard 4-level U-Net for segmentation.
    Input:  (B, C, H, W)
    Output: (B, num_classes, H, W) logits
    """
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))

        x5 = self.bottleneck(self.pool4(x4))

        # Decoder with skip connections
        x = self.up4(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec1(x)

        logits = self.out_conv(x)
        return logits      # again, raw logits


def build_model(model_name: str, in_channels: int, num_classes: int = 1):
    model_name = model_name.lower()
    if model_name == "simple_mlp":
        # Assuming you already have SimpleMLP defined
        return SimpleMLP(in_channels, num_classes)
    elif model_name == "simple_cnn":
        return SimpleCNN(in_channels, num_classes)
    elif model_name in ["unet", "u-net", "unet_cnn"]:
        return UNet(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

