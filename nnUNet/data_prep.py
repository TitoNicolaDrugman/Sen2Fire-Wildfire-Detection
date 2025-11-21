import os
import sys
import json
import numpy as np
import nibabel as nib
import yaml
from pathlib import Path
from utils import kaggle_download_dataset

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NnUNetDatasetPreparator:
    """Class for converting Sen2Fire data to nnU-Net v2 format."""
    
    def __init__(self, data_path, train_scenes, test_scenes, out_dir):
        self.data_path = data_path
        self.train_scenes = train_scenes
        self.test_scenes = test_scenes
        self.out_dir = out_dir
        self.imagesTr = os.path.join(out_dir, "imagesTr")
        self.labelsTr = os.path.join(out_dir, "labelsTr")
        self.imagesTs = os.path.join(out_dir, "imagesTs")
        self.labelsTs = os.path.join(out_dir, "labelsTs")
        self.num_train = 0
        self.num_test = 0
        self.num_modalities = 13
        
    def prepare(self):
        """Converts .npz files to .nii.gz and creates dataset.json."""
        self._convert_data()
        self._create_json()
        return self.num_train, self.num_test
    
    def _convert_data(self):
        """Converts .npz files to nnU-Net format (.nii.gz)."""
        os.makedirs(self.imagesTr, exist_ok=True)
        os.makedirs(self.labelsTr, exist_ok=True)
        os.makedirs(self.imagesTs, exist_ok=True)
        os.makedirs(self.labelsTs, exist_ok=True)
        
        # Train data
        for scene in self.train_scenes:
            self._process_scene(scene, is_train=True)
        
        # Test data
        for scene in self.test_scenes:
            self._process_scene(scene, is_train=False)
        
        print(f"Train: {self.num_train}, Test: {self.num_test}")
    
    def _process_scene(self, scene, is_train):
        """Processes all .npz files in a scene."""
        scene_dir = os.path.join(self.data_path, scene)
        if not os.path.isdir(scene_dir):
            return
        for fname in sorted(os.listdir(scene_dir)):
            if fname.endswith(".npz"):
                self._process_npz(os.path.join(scene_dir, fname), is_train)
    
    def _process_npz(self, npz_path, is_train):
        """Processes a single .npz file."""
        with np.load(npz_path) as data:
            image = data["image"].astype(np.float32)
            aerosol = data["aerosol"].astype(np.float32)
            label = data["label"].astype(np.float32)
        
        all_channels = np.concatenate([image, np.expand_dims(aerosol, axis=0)], axis=0)
        
        # Select directories and case ID based on train/test
        if is_train:
            img_dir, lab_dir = self.imagesTr, self.labelsTr
            case_id = f"Sen2Fire_{self.num_train:04d}"
            self.num_train += 1
        else:
            img_dir, lab_dir = self.imagesTs, self.labelsTs
            case_id = f"Sen2Fire_{self.num_train + self.num_test:04d}"
            self.num_test += 1
        
        affine = np.eye(4, dtype=np.float32)
        
        # Save channels
        for i in range(self.num_modalities):
            channel_data = np.expand_dims(all_channels[i], axis=-1)
            img_path = os.path.join(img_dir, f"{case_id}_{i:04d}.nii.gz")
            nib.save(nib.Nifti1Image(channel_data, affine=affine), img_path)
        
        # Save label
        label_data = np.expand_dims(label, axis=-1)
        lab_path = os.path.join(lab_dir, f"{case_id}.nii.gz")
        nib.save(nib.Nifti1Image(label_data.astype(np.uint8), affine=affine), lab_path)
    
    def _create_json(self):
        """Creates dataset.json for nnU-Net v2."""
        channel_names = {str(i): f"S2_B{i+1:02d}" for i in range(12)}
        channel_names["12"] = "S5P_AEROSOL"
        
        dataset_json = {
            "channel_names": channel_names,
            "labels": {"background": 0, "fire": 1},
            "numTraining": self.num_train,
            "numTest": self.num_test,
            "file_ending": ".nii.gz",
            "dataset_name": os.path.basename(os.path.normpath(self.out_dir)),
            "description": "Sen2Fire wildfire segmentation dataset",
            "reference": "https://arxiv.org/abs/2403.17884",
            "licence": "CC-BY-4.0",
            "release": "1.0",
        }
        
        json_path = os.path.join(self.out_dir, "dataset.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, indent=4, ensure_ascii=False)
        
        print(f"dataset.json saved: {json_path}")


if __name__ == "__main__":
    data_path = kaggle_download_dataset("shariaarfin/sen2fire")
    train_scenes = ['scene1', 'scene2', 'scene3']
    test_scenes = ['scene4']
    out_dir = "nnUNet_raw/Dataset001_Sen2Fire"
    
    preparator = NnUNetDatasetPreparator(data_path, train_scenes, test_scenes, out_dir)
    num_train, num_test = preparator.prepare()
    print(f"✓ Train: {num_train}, Test: {num_test}")