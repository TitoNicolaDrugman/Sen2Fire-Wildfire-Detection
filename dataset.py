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
            scene_names (list): List of scene folders to include (e.g., ['scene1', 'scene2']).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.transform = transform
        self.file_paths = []

        for scene in scene_names:
            scene_dir = os.path.join(data_path, scene)
            if not os.path.isdir(scene_dir):
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

            for filename in os.listdir(scene_dir):
                if filename.endswith('.npz'):
                    self.file_paths.append(os.path.join(scene_dir, filename))
        
        if not self.file_paths:
            raise RuntimeError(f"No .npz files found for scenes: {scene_names}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        
        with np.load(filepath) as data:
            image = data['image'].astype(np.float32)
            aerosol = data['aerosol'].astype(np.float32)
            label = data['label'].astype(np.float32)

        aerosol = np.expand_dims(aerosol, axis=0)
        label = np.expand_dims(label, axis=0)

        input_data = np.concatenate((image, aerosol), axis=0)

        input_tensor = torch.from_numpy(input_data)
        label_tensor = torch.from_numpy(label)
        
        sample = {'input': input_tensor, 'label': label_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample