# augmentations.py

class NoAugmentation:
    """
    A placeholder transform that applies no augmentations.
    Since the original size is 512x512, no resizing is needed.
    """
    def __call__(self, sample):
        # need to add augmentation here in the future
        return sample