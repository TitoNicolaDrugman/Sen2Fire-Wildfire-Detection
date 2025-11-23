# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


# ---------- Attention and basic blocks ----------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Scales feature channels with learned per-channel weights.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class ConvBlock(nn.Module):
    """
    Double conv: (Conv -> BN -> ReLU) x 2, followed by SE channel attention.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels, reduction=16)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)
        return x


class DownBlock(nn.Module):
    """
    Downscale by 2 then apply ConvBlock.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class UpBlock(nn.Module):
    """
    Upsample by 2 with transposed conv to 'out_channels',
    concatenate skip features, then ConvBlock to 'out_channels'.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.dropout = nn.Dropout(0.3) # Add a dropout layer with p=0.3

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed to handle uneven dims
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return self.dropout(x) # Apply dropout after the conv block


# ---------- Transformer bottleneck ----------

class TransformerBottleneck(nn.Module):
    """
    Lightweight Transformer encoder operating on the pooled 2D feature map.
    Expects (B, C=d_model, H=base_hw, W=base_hw) and returns same shape.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=4, base_hw=32, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.base_hw = base_hw

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # PyTorch transformer expects (S, B, E)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learned 2D positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, base_hw, base_hw))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pre_ln = nn.LayerNorm(d_model)
        self.post_proj = nn.Conv2d(d_model, d_model, kernel_size=1)

    """
    def forward(self, x):
        
        # x: (B, C, H, W) with C == d_model and H=W==base_hw
        b, c, h, w = x.shape
        assert c == self.d_model and h == self.base_hw and w == self.base_hw, \
            "TransformerBottleneck: unexpected input shape"

        x = x + self.pos_embed  # (B, C, H, W)
        # to (S, B, E)
        x = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        x = self.pre_ln(x)
        x = self.encoder(x)  # (HW, B, C)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # (B, C, H, W)
        x = self.post_proj(x)
        return x
    """
    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.d_model and h == self.base_hw and w == self.base_hw, "TransformerBottleneck: unexpected input shape"
        orig_dtype = x.dtype
        with autocast('cuda', enabled=False):
            x32 = x.float() + self.pos_embed.float()
            x32 = x32.flatten(2).permute(2, 0, 1)
            x32 = self.pre_ln(x32)
            x32 = self.encoder(x32)
            x32 = x32.permute(1, 2, 0).view(b, c, h, w)
            x32 = self.post_proj(x32)
            return x32.to(orig_dtype)

# ---------- ASPP (multi-scale) ----------

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling with 4 parallel branches and a 1x1 projection.
    Keeps channels at 'out_channels'.
    """
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        c = out_channels // 4
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(4 * c, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.project(x)


# ---------- Models ----------

class SimpleCNN(nn.Module):
    """
    Minimal baseline (kept for compatibility with get_model).
    """
    def __init__(self, input_channels=13, output_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
        )
        self.decoder = nn.Sequential(
            UpBlock(512, 256, 256),
            UpBlock(256, 128, 128),
            UpBlock(128, 64, 64),
        )
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)

        u1 = self.decoder[0](x4, x3)
        u2 = self.decoder[1](u1, x2)
        u3 = self.decoder[2](u2, x1)
        return self.out_conv(u3)


class CNNTransformerUNet(nn.Module):
    """
    UNet encoder/decoder with a pooled Transformer bottleneck and ASPP,
    using SE-enhanced ConvBlocks throughout.
    """
    def __init__(self, input_channels=13, output_channels=1):
        super().__init__()
        # Encoder
        self.in_conv = ConvBlock(input_channels, 64)   # 512x512 -> 64c
        self.down1   = DownBlock(64, 128)              # 256x256
        self.down2   = DownBlock(128, 256)             # 128x128
        self.down3   = DownBlock(256, 512)             # 64x64

        # Bottleneck: pool to 32x32, transformer, ASPP, then upsample to 64x64
        self.bottleneck_pool = nn.MaxPool2d(2)  # 64x64 -> 32x32
        self.transformer = TransformerBottleneck(d_model=512, nhead=8, num_layers=4, base_hw=32)
        self.aspp = ASPP(512, 512)
        self.bottleneck_up = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 32 -> 64

        # Decoder
        self.up1 = UpBlock(in_channels=512,  skip_channels=512, out_channels=512)  # 64
        self.up2 = UpBlock(in_channels=512,  skip_channels=256, out_channels=256)  # 128
        self.up3 = UpBlock(in_channels=256,  skip_channels=128, out_channels=128)  # 256
        self.up4 = UpBlock(in_channels=128,  skip_channels=64,  out_channels=64)   # 512

        # Output head
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)   # 64 @ 512
        x2 = self.down1(x1)    # 128 @ 256
        x3 = self.down2(x2)    # 256 @ 128
        x4 = self.down3(x3)    # 512 @ 64

        # Bottleneck
        pooled = self.bottleneck_pool(x4)                 # 512 @ 32
        transformer_out = self.transformer(pooled)        # 512 @ 32
        transformer_out = self.aspp(transformer_out)      # 512 @ 32 (multi-scale)
        upsampled = self.bottleneck_up(transformer_out)   # 512 @ 64

        # Decoder with skips
        u1 = self.up1(upsampled, x4)  # 512 @ 64
        u2 = self.up2(u1, x3)         # 256 @ 128
        u3 = self.up3(u2, x2)         # 128 @ 256
        u4 = self.up4(u3, x1)         # 64  @ 512

        return self.out_conv(u4)


# ---------- Factory ----------

def get_model(model_name: str, **kwargs):
    """
    Factory function to get a model instance by name.
    """
    models = {
        "SimpleCNN": SimpleCNN,
        "CNNTransformerUNet": CNNTransformerUNet,
    }
    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(models.keys())}")
    print(f"Initializing model: {model_name}")
    return model_class(**kwargs)
