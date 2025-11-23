# augmentations.py
import torch
import random
from torchvision import transforms

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
    
class RandomRotate90:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, sample):
        import random, torch
        if random.random() < self.p:
            k = int(random.choice([1,2,3]))
            sample['input'] = torch.rot90(sample['input'], k, dims=[1,2])
            sample['label'] = torch.rot90(sample['label'], k, dims=[1,2])
        return sample

# --- NEW AUGMENTATIONS ---
class RandomColorJitter:
    """Applies random color jittering to the input image."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample):
        if random.random() < self.p:
            # ColorJitter expects image in range [0, 1] and of shape (3, H, W)
            # We will apply it to the RGB bands (indices 3, 2, 1) after normalizing them
            img = sample['input']
            rgb_bands = img[[3, 2, 1], :, :]

            # Simple min-max normalization for the RGB bands for jittering
            min_val = rgb_bands.min()
            max_val = rgb_bands.max()
            if max_val > min_val:
                rgb_bands = (rgb_bands - min_val) / (max_val - min_val)
                rgb_bands_jittered = self.jitter(rgb_bands)
                # Denormalize back to original range
                rgb_bands_jittered = rgb_bands_jittered * (max_val - min_val) + min_val
                # Replace the original bands
                img[[3, 2, 1], :, :] = rgb_bands_jittered
            
            sample['input'] = img
        return sample

class RandomGaussianBlur:
    """Applies Gaussian blur to the input image."""
    def __init__(self, kernel_size=3, p=0.3):
        self.p = p
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size)

    def __call__(self, sample):
        if random.random() < self.p:
            sample['input'] = self.blur(sample['input'])
        return sample