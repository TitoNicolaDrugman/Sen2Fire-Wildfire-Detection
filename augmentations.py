# augmentations.py

class Standardize:
    """
    Applies Z-score standardization to the input tensor using pre-computed
    channel-wise mean and standard deviation.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        normalized_input = (sample['input'] - self.mean) / (self.std + 1e-6) # Add epsilon to avoid division by zero
        sample['input'] = normalized_input
        return sample

class NoAugmentation:
    """
    A placeholder transform that applies no augmentations.
    Since the original size is 512x512, no resizing is needed.
    """
    def __call__(self, sample):
        # need to add augmentation here in the future
        return sample