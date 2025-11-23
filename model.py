# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class DualStreamSegFormer(nn.Module):
    """
    Dual-Stream SegFormer with Feature-Level Fusion.
    Stream 1 (Expert): Receives high-signal bands (SWIR, NIR, Aerosol).
    Stream 2 (Context): Receives noisy/texture bands (Visual, Vegetation).
    """
    def __init__(self, input_channels=13, output_channels=1, 
                 expert_indices=[11, 7, 3, 12], #[11, 10, 7, 3, 12], # B12, B11, B8, B4, Aerosol
                 context_indices=[0, 1, 2, 4, 5, 6, 8, 9, 10]): # The rest
        super().__init__()
        
        self.expert_indices = expert_indices
        self.context_indices = context_indices
        
        # --- 1. Define Encoders (MiT-B0) ---
        # We instantiate two full models but only use their encoders
        self.expert_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=output_channels, ignore_mismatched_sizes=True
        )
        self.context_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=output_channels, ignore_mismatched_sizes=True
        )
        
        # --- 2. Adapt Input Layers ---
        # Modify Expert First Layer
        self._modify_first_layer(self.expert_model, len(expert_indices))
        # Modify Context First Layer
        self._modify_first_layer(self.context_model, len(context_indices))

        # --- 3. Fusion Layers ---
        # SegFormer encoders (MiT-B0) output features at channels: [32, 64, 160, 256]
        # We concatenate features from both streams, so channels double.
        # We use 1x1 convs to project them back to original size for the decoder.
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(32*2, 32, kernel_size=1),
            nn.Conv2d(64*2, 64, kernel_size=1),
            nn.Conv2d(160*2, 160, kernel_size=1),
            nn.Conv2d(256*2, 256, kernel_size=1)
        ])
        
        # --- 4. Decoder ---
        # We use the decoder head from the expert model
        self.decoder = self.expert_model.decode_head

    def _modify_first_layer(self, model, n_channels):
        """Helper to replace the first convolution layer."""
        original_layer = model.segformer.encoder.patch_embeddings[0].proj
        new_layer = nn.Conv2d(
            in_channels=n_channels,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding
        )
        # Initialize: Copy weights from RGB channels for the first 3, random for others
        with torch.no_grad():
            new_layer.weight[:, :3, :, :] = original_layer.weight[:, :3, :, :]
            if n_channels > 3:
                nn.init.kaiming_normal_(new_layer.weight[:, 3:, :, :])
        
        model.segformer.encoder.patch_embeddings[0].proj = new_layer

    def forward(self, x):
        # x shape: (B, 13, 512, 512)
        
        # 1. Split Inputs
        x_expert = x[:, self.expert_indices, :, :]
        x_context = x[:, self.context_indices, :, :]
        
        # 2. Encode Streams
        # output_hidden_states=True gives us the features at 4 scales
        out_expert = self.expert_model.segformer.encoder(x_expert, output_hidden_states=True)
        out_context = self.context_model.segformer.encoder(x_context, output_hidden_states=True)
        
        feats_expert = out_expert.hidden_states # Tuple of 4 tensors
        feats_context = out_context.hidden_states
        
        # 3. Fuse Features
        fused_features = []
        for i in range(len(feats_expert)):
            # Concatenate along channel dimension (dim=1)
            cat_feat = torch.cat([feats_expert[i], feats_context[i]], dim=1)
            # Project back to original channel size
            projected_feat = self.fusion_convs[i](cat_feat)
            fused_features.append(projected_feat)
            
        # 4. Decode
        logits = self.decoder(fused_features)
        
        # 5. Upsample to input size (512x512)
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        return upsampled_logits

# --- Single Stream Wrapper (Kept for backward compatibility) ---
class SegFormer(nn.Module):
    def __init__(self, input_channels=13, output_channels=1):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=output_channels,
            ignore_mismatched_sizes=True
        )
        original_first_layer = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding
        )
        nn.init.kaiming_normal_(self.model.segformer.encoder.patch_embeddings[0].proj.weight)

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return F.interpolate(outputs.logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

# --- Factory ---
def get_model(model_name: str, **kwargs):
    models = {
        "SegFormer": SegFormer,
        "DualStreamSegFormer": DualStreamSegFormer, # Added new model
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    print(f"Initializing model: {model_name}")
    return models[model_name](**kwargs)