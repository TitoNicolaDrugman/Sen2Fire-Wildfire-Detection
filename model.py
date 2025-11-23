# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation


# ---------- NEW SEGFORMER MODEL WRAPPER ----------

class SegFormer(nn.Module):
    """
    Wrapper for the Hugging Face SegformerForSemanticSegmentation model,
    modified to accept a different number of input channels and output a single channel.
    """
    def __init__(self, input_channels=13, output_channels=1): # output_channels is now used
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=output_channels, # MODIFIED: Use output_channels (which will be 1)
            ignore_mismatched_sizes=True
        )

        # --- MODIFICATION START ---
        # The original first convolution layer in the mit-b0 backbone
        original_first_layer = self.model.segformer.encoder.patch_embeddings[0].proj
        
        # Create a new convolution layer with the desired number of input channels
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding
        )
        # Initialize the new layer's weights (e.g., using Kaiming normal)
        nn.init.kaiming_normal_(self.model.segformer.encoder.patch_embeddings[0].proj.weight, mode='fan_in', nonlinearity='relu')
        print(f"SegFormer's first conv layer modified to accept {input_channels} channels.")
        # --- MODIFICATION END ---

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        # Upsample logits to match original size
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return upsampled_logits

# ... (The rest of the file remains the same, it is not used by this model) ...

# ---------- Factory ----------
def get_model(model_name: str, **kwargs):
    """
    Factory function to get a model instance by name.
    """
    models = {
        "SimpleCNN": SimpleCNN,
        "CNNTransformerUNet": CNNTransformerUNet,
        "SegFormer": SegFormer,
    }
    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(models.keys())}")
    print(f"Initializing model: {model_name}")
    kwargs.pop('important_channel_indices', None)
    return model_class(**kwargs)



# The rest of the original model.py code (SEBlock, ConvBlock, etc.) can remain here
# as they are not used by the SegFormer model but are part of the get_model factory.
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // reduction, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channels // reduction, channels, 1, bias=True), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.avg(x))
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.se = SEBlock(out_channels, reduction=16)
    def forward(self, x):
        return self.se(self.double_conv(x))
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))
    def forward(self, x):
        return self.encoder(x)
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diffY, diffX = skip.size(-2) - x.size(-2), skip.size(-1) - x.size(-1)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([skip, x], dim=1))
class TransformerBottleneck(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, base_hw=32, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model, self.base_hw = d_model, base_hw
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, base_hw, base_hw))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pre_ln = nn.LayerNorm(d_model)
        self.post_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.d_model and h == self.base_hw and w == self.base_hw, "TransformerBottleneck: unexpected input shape"
        orig_dtype = x.dtype
        with autocast('cuda', enabled=False):
            x32 = (x.float() + self.pos_embed.float()).flatten(2).permute(2, 0, 1)
            x32 = self.pre_ln(x32)
            x32 = self.encoder(x32)
            x32 = x32.permute(1, 2, 0).view(b, c, h, w)
            x32 = self.post_proj(x32)
            return x32.to(orig_dtype)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        c = out_channels // 4
        self.b1, self.b2, self.b3, self.b4 = (nn.Sequential(nn.Conv2d(in_channels, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)), nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=rates[1], dilation=rates[1], bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)), nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=rates[2], dilation=rates[2], bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)), nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=rates[3], dilation=rates[3], bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)))
        self.project = nn.Sequential(nn.Conv2d(4 * c, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.project(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=13, output_channels=1):
        super().__init__()
        self.encoder, self.decoder = (nn.Sequential(ConvBlock(input_channels, 64), DownBlock(64, 128), DownBlock(128, 256), DownBlock(256, 512)), nn.Sequential(UpBlock(512, 256, 256), UpBlock(256, 128, 128), UpBlock(128, 64, 64)))
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder[0](x), self.encoder[1](x1), self.encoder[2](x2), self.encoder[3](x3)
        return self.out_conv(self.decoder[2](self.decoder[1](self.decoder[0](x4, x3), x2), x1))
class CNNTransformerUNet(nn.Module):
    def __init__(self, input_channels=13, output_channels=1):
        super().__init__()
        self.in_conv, self.down1, self.down2, self.down3 = (ConvBlock(input_channels, 64), DownBlock(64, 128), DownBlock(128, 256), DownBlock(256, 512))
        self.bottleneck_pool, self.transformer, self.aspp, self.bottleneck_up = (nn.MaxPool2d(2), TransformerBottleneck(d_model=512, nhead=8, num_layers=4, base_hw=32), ASPP(512, 512), nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2))
        self.up1, self.up2, self.up3, self.up4 = (UpBlock(in_channels=512, skip_channels=512, out_channels=512), UpBlock(in_channels=512, skip_channels=256, out_channels=256), UpBlock(in_channels=256, skip_channels=128, out_channels=128), UpBlock(in_channels=128, skip_channels=64, out_channels=64))
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)
    def forward(self, x):
        x1, x2, x3, x4 = self.in_conv(x), self.down1(x1), self.down2(x2), self.down3(x3)
        upsampled = self.bottleneck_up(self.aspp(self.transformer(self.bottleneck_pool(x4))))
        return self.out_conv(self.up4(self.up3(self.up2(self.up1(upsampled, x4), x3), x2), x1))