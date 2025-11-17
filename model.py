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

# ---------------------------
# UNet implementation below
# ---------------------------

class ConvBlock(nn.Module):
    """(Conv => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """Downscaling with MaxPool then ConvBlock"""
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, **kwargs)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

# --- Replace Up and UNet with these corrected implementations ---

class Up(nn.Module):
    """
    Upscaling then ConvBlock.
    Constructor now expects explicit channel counts:
      decoder_in_ch: number of channels in the decoder input (the 'x' passed to forward)
      skip_ch: number of channels in the skip connection
      out_ch: number of output channels produced by this block (also the decoder channel for next block)
    Behavior:
      1) ConvTranspose2d(decoder_in_ch -> out_ch) upsamples spatially and reduces channels
      2) Concatenate with skip (out_ch + skip_ch)
      3) ConvBlock(out_ch + skip_ch -> out_ch)
    """
    def __init__(self, decoder_in_ch, skip_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        # upsample decoder features from decoder_in_ch -> out_ch
        self.up = nn.ConvTranspose2d(decoder_in_ch, out_ch, kernel_size=2, stride=2)
        # after concat: channels = out_ch + skip_ch -> produce out_ch
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, use_bn=use_bn, dropout=dropout)

    def forward(self, x, skip):
        """
        x: decoder feature (B, decoder_in_ch, H, W)
        skip: encoder feature (B, skip_ch, H*2, W*2)
        """
        x = self.up(x)  # (B, out_ch, H*2, W*2)
        # Crop skip if required
        if x.shape[-2:] != skip.shape[-2:]:
            skip = self._center_crop(skip, x.shape[-2:])
        x = torch.cat([skip, x], dim=1)  # (B, out_ch + skip_ch, H*2, W*2)
        x = self.conv(x)                 # (B, out_ch, H*2, W*2)
        return x

    @staticmethod
    def _center_crop(x, target_hw):
        _, _, h, w = x.shape
        th, tw = target_hw
        dh = (h - th) // 2
        dw = (w - tw) // 2
        return x[:, :, dh:dh+th, dw:dw+tw]


class UNet(nn.Module):
    """
    Corrected U-Net using the new Up signature.
    """
    def __init__(self, input_channels=3, n_classes=1, base_filters=32, use_bn=True, dropout=0.0):
        super().__init__()
        f = base_filters

        # Encoder
        self.inc = ConvBlock(input_channels, f, use_bn, dropout)      # f
        self.down1 = Down(f, f*2, use_bn=use_bn, dropout=dropout)     # 2f
        self.down2 = Down(f*2, f*4, use_bn=use_bn, dropout=dropout)   # 4f
        self.down3 = Down(f*4, f*8, use_bn=use_bn, dropout=dropout)   # 8f
        self.down4 = Down(f*8, f*16, use_bn=use_bn, dropout=dropout)  # 16f

        # Bottleneck
        self.bottleneck = ConvBlock(f*16, f*16, use_bn=use_bn, dropout=dropout)  # 16f

        # Decoder: provide explicit channel counts (decoder_in_ch, skip_ch, out_ch)
        # d4: decoder_in_ch = 16f (from bottleneck), skip_ch = 8f -> out_ch = 8f
        self.up4 = Up(decoder_in_ch=f*16, skip_ch=f*8, out_ch=f*8, use_bn=use_bn, dropout=dropout)
        # d3: decoder_in_ch = 8f, skip_ch = 4f -> out_ch = 4f
        self.up3 = Up(decoder_in_ch=f*8, skip_ch=f*4, out_ch=f*4, use_bn=use_bn, dropout=dropout)
        # d2: decoder_in_ch = 4f, skip_ch = 2f -> out_ch = 2f
        self.up2 = Up(decoder_in_ch=f*4, skip_ch=f*2, out_ch=f*2, use_bn=use_bn, dropout=dropout)
        # d1: decoder_in_ch = 2f, skip_ch = f -> out_ch = f
        self.up1 = Up(decoder_in_ch=f*2, skip_ch=f, out_ch=f, use_bn=use_bn, dropout=dropout)

        self.final_conv = nn.Conv2d(f, n_classes, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # (B, f, H, W)
        x2 = self.down1(x1)   # (B, 2f, H/2, W/2)
        x3 = self.down2(x2)   # (B, 4f, H/4, W/4)
        x4 = self.down3(x3)   # (B, 8f, H/8, W/8)
        x5 = self.down4(x4)   # (B, 16f, H/16, W/16)

        # Bottleneck
        b = self.bottleneck(x5)  # (B, 16f, H/16, W/16)

        # Decoder (use corresponding skip of same spatial size)
        d4 = self.up4(b, x4)  # (B, 8f, H/8, W/8)
        d3 = self.up3(d4, x3) # (B, 4f, H/4, W/4)
        d2 = self.up2(d3, x2) # (B, 2f, H/2, W/2)
        d1 = self.up1(d2, x1) # (B, f, H, W)

        out = self.final_conv(d1)
        return out  # raw logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------
# Factory
# ---------------------------
def get_model(model_name: str = "simple_cnn", input_channels: int = 3, n_classes: int = 1, **kwargs):
    """
    Factory: supports:
      - "simple_mlp" -> SimpleMLP (expects input_channels arg)
      - "simple_cnn" / "cnn" / "conv" -> SimpleCNN
      - "unet" / "unet_small" -> UNet

    Additional kwargs used by UNet: base_filters (int), use_bn (bool), dropout (float)
    """
    mn = model_name.lower()
    if mn in ("simple_mlp", "mlp"):
        return SimpleMLP(input_channels, output_channels=n_classes)
    if mn in ("simple_cnn", "cnn", "conv"):
        return SimpleCNN(input_channels=input_channels, n_classes=n_classes)
    if mn in ("unet", "unet_small", "unet_simple"):
        return UNet(input_channels=input_channels,
                    n_classes=n_classes,
                    base_filters=kwargs.get('base_filters', 32),
                    use_bn=kwargs.get('use_bn', True),
                    dropout=kwargs.get('dropout', 0.0))
    raise ValueError(f"Unknown model_name: {model_name}")
