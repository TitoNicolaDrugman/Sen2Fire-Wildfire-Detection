
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Sen2FireDataset(Dataset):
    """
    Custom PyTorch Dataset for the Sen2Fire dataset.
    Reads image, aerosol, and label data from .npz files and builds
    different input strategies (RGB, SWIR, NBR, NDVI, etc. + aerosol).
    """

    def __init__(self, data_path, scene_names, transform=None, input_strategy="vanilla"):
        """
        Args:
            data_path (str): Path to the main dataset directory.
            scene_names (list): List of scene folders to include.
            transform (callable, optional): Optional transform to be applied on a sample.
            input_strategy (str): one of
                - 'rgb_aerosol'
                - 'swir_aerosol'
                - 'nbr_aerosol'
                - 'ndvi_aerosol'
                - 'rgb_swir_nbr_ndvi_aerosol'
                - 'vanilla'
        """
        super().__init__()
        self.transform = transform
        self.input_strategy = input_strategy
        self.file_paths = []

        for scene in scene_names:
            scene_dir = os.path.join(data_path, scene)
            if not os.path.isdir(scene_dir):
                print(f"Warning: Scene directory not found: {scene_dir}")
                continue

            for filename in os.listdir(scene_dir):
                if filename.endswith('.npz'):
                    self.file_paths.append(os.path.join(scene_dir, filename))
        
        if not self.file_paths:
            raise RuntimeError(f"No .npz files found for scenes: {scene_names} in path {data_path}")

    def __len__(self):
        return len(self.file_paths)

    def _build_input(self, full_input: np.ndarray) -> np.ndarray:
        """
        full_input: (13, H, W) = [B1..B12, B13(aerosol)]
        Returns: (C', H, W) according to input_strategy.
        """

        # Band indices (0-based) after concatenation image(12) + aerosol(1)
        B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13 = range(13)

        s = self.input_strategy.lower()

        if s == "rgb_aerosol":
            # B2 (Blue), B3 (Green), B4 (Red) + aerosol index
            idx = [B2, B3, B4, B13]
            return full_input[idx, :, :]

        elif s == "swir_aerosol":
            # SWIR composite + aerosol: use B11, B12 as SWIR bands + aerosol
            idx = [B11, B12, B13]  # 3 channels
            return full_input[idx, :, :]

        elif s == "nbr_aerosol":
            # NBR = (NIR - SWIR2) / (NIR + SWIR2) using B8 (NIR) and B12 (SWIR)
            nir = full_input[B8]   # (H, W)
            swir2 = full_input[B12]
            denom = nir + swir2
            nbr = np.zeros_like(nir, dtype=np.float32)
            mask = denom != 0
            nbr[mask] = (nir[mask] - swir2[mask]) / denom[mask]
            nbr = np.expand_dims(nbr, axis=0)  # (1, H, W)
            aerosol = full_input[B13:B13+1]    # (1, H, W)
            return np.concatenate([nbr, aerosol], axis=0)  # (2, H, W)

        elif s == "ndvi_aerosol":
            # NDVI = (NIR - Red) / (NIR + Red) using B8 (NIR) and B4 (Red)
            nir = full_input[B8]
            red = full_input[B4]
            denom = nir + red
            ndvi = np.zeros_like(nir, dtype=np.float32)
            mask = denom != 0
            ndvi[mask] = (nir[mask] - red[mask]) / denom[mask]
            ndvi = np.expand_dims(ndvi, axis=0)  # (1, H, W)
            aerosol = full_input[B13:B13+1]      # (1, H, W)
            return np.concatenate([ndvi, aerosol], axis=0)  # (2, H, W)

        elif s == "rgb_swir_nbr_ndvi_aerosol":
            # Combine everything:
            #   RGB: B2,B3,B4
            #   SWIR: B11,B12
            #   NBR: from B8,B12
            #   NDVI: from B8,B4
            #   Aerosol: B13
            rgb = full_input[[B2, B3, B4], :, :]          # (3, H, W)
            swir = full_input[[B11, B12], :, :]           # (2, H, W)

            nir = full_input[B8]
            red = full_input[B4]
            swir2 = full_input[B12]

            # NBR
            denom_nbr = nir + swir2
            nbr = np.zeros_like(nir, dtype=np.float32)
            mask_nbr = denom_nbr != 0
            nbr[mask_nbr] = (nir[mask_nbr] - swir2[mask_nbr]) / denom_nbr[mask_nbr]
            nbr = np.expand_dims(nbr, axis=0)             # (1, H, W)

            # NDVI
            denom_ndvi = nir + red
            ndvi = np.zeros_like(nir, dtype=np.float32)
            mask_ndvi = denom_ndvi != 0
            ndvi[mask_ndvi] = (nir[mask_ndvi] - red[mask_ndvi]) / denom_ndvi[mask_ndvi]
            ndvi = np.expand_dims(ndvi, axis=0)           # (1, H, W)

            aerosol = full_input[B13:B13+1]               # (1, H, W)

            return np.concatenate([rgb, swir, nbr, ndvi, aerosol], axis=0)
            # total channels = 3 + 2 + 1 + 1 + 1 = 8

        elif s == "vanilla":
            # Use all 13 bands directly (B1..B13)
            return full_input

        else:
            raise ValueError(f"Unknown input_strategy: {self.input_strategy}")

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        
        with np.load(filepath) as data:
            image = data['image'].astype(np.float32)      # (12, 512, 512)
            aerosol = data['aerosol'].astype(np.float32)  # (512, 512) -> B13
            label = data['label'].astype(np.float32)      # (512, 512)

        aerosol = np.expand_dims(aerosol, axis=0)         # (1, 512, 512)
        label   = np.expand_dims(label,   axis=0)         # (1, 512, 512)

        # full_input: (13, H, W)
        full_input = np.concatenate((image, aerosol), axis=0)

        # Build according to strategy
        input_data = self._build_input(full_input)        # (C', H, W)

        input_tensor = torch.from_numpy(input_data)
        label_tensor = torch.from_numpy(label)

        sample = {'input': input_tensor, 'label': label_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample
