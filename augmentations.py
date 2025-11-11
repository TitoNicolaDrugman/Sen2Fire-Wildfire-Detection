# augmentations.py
import torch
import random

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class Standardize:
    """Applies Z-score standardization."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # Add epsilon to std to avoid division by zero
        sample['input'] = (sample['input'] - self.mean) / (self.std + 1e-6)
        return sample

class RandomHorizontalFlip:
    """Randomly flips the image and mask horizontally."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['input'] = torch.flip(sample['input'], [2])
            sample['label'] = torch.flip(sample['label'], [2])
        return sample

class RandomVerticalFlip:
    """Randomly flips the image and mask vertically."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['input'] = torch.flip(sample['input'], [1])
            sample['label'] = torch.flip(sample['label'], [1])
        return sample