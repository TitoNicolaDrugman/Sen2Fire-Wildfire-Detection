# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Sen2FireDataset(Dataset):
    """
    Custom PyTorch Dataset for the Sen2Fire dataset.
    Reads image, aerosol, and label data from .npz files.
    """
    def __init__(self, data_path, scene_names, transform=None):
        """
        Args:
            data_path (str): Path to the main dataset directory.
            scene_names (list): List of scene folders to include.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.transform = transform
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

    def __getitem__(self, idx):
            filepath = self.file_paths[idx]
            
            with np.load(filepath) as data:
                image = data['image'].astype(np.float32)      # Shape (12, 512, 512)
                aerosol = data['aerosol'].astype(np.float32)  # Shape (512, 512)
                label = data['label'].astype(np.float32)      # Shape (512, 512)

            # Reshape aerosol and label to have a channel dimension
            aerosol = np.expand_dims(aerosol, axis=0) # -> (1, 512, 512)
            label = np.expand_dims(label, axis=0)     # -> (1, 512, 512)

            # Concatenate sensor bands and aerosol to form the input
            input_data = np.concatenate((image, aerosol), axis=0) # -> (13, 512, 512)

            # --- START MODIFICATION ---
            # Keep only B12 (idx 11), B8 (idx 7), B4 (idx 3), and Aerosol (idx 12)
            # The order will be: [B12, B8, B4, Aerosol]
            desired_indices = [11, 7, 3, 12]
            input_data = input_data[desired_indices, :, :]
            # --- END MODIFICATION ---

            # Convert to PyTorch tensors
            input_tensor = torch.from_numpy(input_data)
            label_tensor = torch.from_numpy(label)
            
            # Package into a dictionary
            sample = {'input': input_tensor, 'label': label_tensor}

            # Apply transformations
            if self.transform:
                sample = self.transform(sample)

            return sample