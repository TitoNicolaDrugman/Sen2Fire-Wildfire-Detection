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

class GaussianNoise:
    """Adds Gaussian noise to the input."""
    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            noise = torch.randn_like(sample['input']) * self.std
            sample['input'] = sample['input'] + noise
        return sample

class RandomBrightnessContrast:
    """Randomly adjusts brightness and contrast."""
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            brightness = random.uniform(-self.brightness_limit, self.brightness_limit)
            contrast = random.uniform(1 - self.contrast_limit, 1 + self.contrast_limit)
            sample['input'] = sample['input'] * contrast + brightness
        return sample