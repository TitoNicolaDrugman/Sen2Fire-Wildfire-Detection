# model.py
import torch
import torch.nn as nn

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