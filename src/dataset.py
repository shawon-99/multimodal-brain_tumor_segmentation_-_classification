"""
PyTorch Dataset classes for Brain Tumor Classification and Segmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from src.preprocessing import zscore_normalize, apply_augmentation_pipeline
import random


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI images.
    Loads preprocessed images and applies transformations.
    """
    
    def __init__(
        self,
        metadata_csv,
        transform=None,
        augment=False,
        augment_config=None,
        img_size=224,
        return_domain=False
    ):
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        self.augment = augment
        self.augment_config = augment_config
        self.img_size = img_size
        self.return_domain = return_domain
        
        # Create label mapping
        self.classes = sorted(self.metadata['class_name'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Create domain mapping if needed
        if return_domain:
            self.domains = self._extract_domains()
            self.domain_to_idx = {domain: idx for idx, domain in enumerate(sorted(self.domains))}
    
    def _extract_domains(self):
        """Extract domain labels from file paths."""
        domains = set()
        for path in self.metadata['file_path']:
            path_parts = Path(path).parts
            if 'Brain_Tumor_MRI_Dataset' in path_parts:
                domains.add('Brain_Tumor_MRI_Dataset')
            elif 'Brain_Tumor_MRI_Scans' in path_parts:
                domains.add('Brain_Tumor_MRI_Scans')
            elif 'BRATs_2020' in path_parts or 'BraTS' in path_parts:
                domains.add('BRATs_2020')
        return domains
    
    def _get_domain_label(self, file_path):
        """Get domain label from file path."""
        path_parts = Path(file_path).parts
        if 'Brain_Tumor_MRI_Dataset' in path_parts:
            return self.domain_to_idx['Brain_Tumor_MRI_Dataset']
        elif 'Brain_Tumor_MRI_Scans' in path_parts:
            return self.domain_to_idx['Brain_Tumor_MRI_Scans']
        elif 'BRATs_2020' in path_parts or 'BraTS' in path_parts:
            return self.domain_to_idx['BRATs_2020']
        return 0
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get image path and label
        row = self.metadata.iloc[idx]
        img_path = row['file_path']
        class_name = row['class_name']
        label = self.class_to_idx[class_name]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Apply augmentation if training
        if self.augment:
            image = apply_augmentation_pipeline(image, self.augment_config)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Z-score normalization
        img_normalized = zscore_normalize(img_array, per_channel=True)
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        
        # Apply custom transform if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        if self.return_domain:
            domain_label = self._get_domain_label(img_path)
            return img_tensor, label, domain_label
        
        return img_tensor, label
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalance."""
        class_counts = self.metadata['class_name'].value_counts()
        total = len(self.metadata)
        weights = torch.zeros(len(self.classes))
        
        for cls, idx in self.class_to_idx.items():
            if cls in class_counts:
                weights[idx] = total / (len(self.classes) * class_counts[cls])
        
        return weights


class MultiModalBrainTumorDataset(Dataset):
    """
    Dataset for multi-modal MRI (T1, T2, FLAIR, T1-CE).
    Used for BRATs dataset.
    """
    
    def __init__(
        self,
        metadata_csv,
        modalities=['T1', 'T2', 'FLAIR', 'T1-CE'],
        transform=None,
        img_size=224
    ):
        self.metadata = pd.read_csv(metadata_csv)
        self.modalities = modalities
        self.transform = transform
        self.img_size = img_size
        
        # Assuming metadata has columns: T1_path, T2_path, FLAIR_path, T1CE_path, label
        self.modality_columns = {
            'T1': 'T1_path',
            'T2': 'T2_path',
            'FLAIR': 'FLAIR_path',
            'T1-CE': 'T1CE_path'
        }
    
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load all modalities
        modality_tensors = []
        for modality in self.modalities:
            col_name = self.modality_columns[modality]
            img_path = row[col_name]
            
            # Load and preprocess
            image = Image.open(img_path).convert('L')  # Grayscale
            if image.size != (self.img_size, self.img_size):
                image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
            
            img_array = np.array(image)
            img_normalized = zscore_normalize(img_array, per_channel=False)
            img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).float()
            modality_tensors.append(img_tensor)
        
        # Stack modalities (4, H, W)
        multi_modal_tensor = torch.cat(modality_tensors, dim=0)
        
        # Get label
        label = int(row['label'])
        
        if self.transform:
            multi_modal_tensor = self.transform(multi_modal_tensor)
        
        return multi_modal_tensor, label


def create_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    batch_size=32,
    num_workers=4,
    augment_train=True,
    return_domain=False,
    img_size=224
):
    """
    Create train, validation, and test dataloaders.
    """
    # Augmentation config for training
    augment_config = {
        'rotation': True,
        'flip': True,
        'zoom': True,
        'intensity': True,
        'elastic': False
    } if augment_train else None
    
    # Create datasets
    train_dataset = BrainTumorDataset(
        train_csv,
        augment=augment_train,
        augment_config=augment_config,
        img_size=img_size,
        return_domain=return_domain
    )
    
    val_dataset = BrainTumorDataset(
        val_csv,
        augment=False,
        img_size=img_size,
        return_domain=return_domain
    )
    
    test_dataset = BrainTumorDataset(
        test_csv,
        augment=False,
        img_size=img_size,
        return_domain=return_domain
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def get_class_weights(train_csv):
    """Get class weights for handling imbalance."""
    dataset = BrainTumorDataset(train_csv)
    return dataset.get_class_weights()


class BraTSSegmentationDataset(Dataset):
    """
    PyTorch Dataset for BRATs segmentation with multi-modal MRI.
    Loads preprocessed 4-channel images (T1, T1-CE, T2, FLAIR) and segmentation masks.
    """
    
    def __init__(
        self,
        metadata_csv,
        data_root="Dataset/brats_preprocessed/slices",
        augment=False,
        augment_config=None,
        img_size=224,
        return_patient_id=False
    ):
        """
        Args:
            metadata_csv: Path to CSV with columns: patient_id, slice_idx, t1, t1ce, t2, flair, seg, has_tumor
            data_root: Root directory where preprocessed slices are stored
            augment: Whether to apply data augmentation
            augment_config: Augmentation configuration dict
            img_size: Target image size (assumes square images)
            return_patient_id: Whether to return patient ID (for evaluation)
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.data_root = Path(data_root)
        self.augment = augment
        self.augment_config = augment_config or {}
        self.img_size = img_size
        self.return_patient_id = return_patient_id
        
        print(f"Loaded {len(self.metadata)} slices from {metadata_csv}")
        print(f"Slices with tumor: {self.metadata['has_tumor'].sum()}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load all modalities (assuming .npy files from preprocessing)
        t1 = np.load(self.data_root / row['t1'])
        t1ce = np.load(self.data_root / row['t1ce'])
        t2 = np.load(self.data_root / row['t2'])
        flair = np.load(self.data_root / row['flair'])
        seg = np.load(self.data_root / row['seg'])
        
        # Stack modalities into 4-channel image (C, H, W)
        image = np.stack([t1, t1ce, t2, flair], axis=0)  # Shape: (4, H, W)
        
        # Apply augmentation if training
        if self.augment:
            image, seg = self._apply_augmentation(image, seg)
        
        # Resize if needed
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = self._resize_image(image)
            seg = self._resize_mask(seg)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        seg_tensor = torch.from_numpy(seg).long()
        
        if self.return_patient_id:
            return image_tensor, seg_tensor, row['patient_id']
        
        return image_tensor, seg_tensor
    
    def _apply_augmentation(self, image, mask):
        """Apply synchronized augmentation to image and mask."""
        # Random horizontal flip
        if self.augment_config.get('flip', True) and random.random() > 0.5:
            image = np.flip(image, axis=2).copy()  # Flip width dimension
            mask = np.flip(mask, axis=1).copy()
        
        # Random vertical flip
        if self.augment_config.get('flip', True) and random.random() > 0.5:
            image = np.flip(image, axis=1).copy()  # Flip height dimension
            mask = np.flip(mask, axis=0).copy()
        
        # Random rotation (90, 180, 270 degrees)
        if self.augment_config.get('rotation', True):
            k = random.randint(0, 3)  # Number of 90-degree rotations
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
        
        # Intensity variations (only for image, not mask)
        if self.augment_config.get('intensity', True):
            # Random brightness adjustment
            brightness_factor = random.uniform(0.9, 1.1)
            image = image * brightness_factor
            
            # Random contrast adjustment per channel
            for c in range(image.shape[0]):
                if random.random() > 0.5:
                    contrast_factor = random.uniform(0.9, 1.1)
                    mean = image[c].mean()
                    image[c] = (image[c] - mean) * contrast_factor + mean
        
        return image, mask
    
    def _resize_image(self, image):
        """Resize multi-channel image."""
        from scipy.ndimage import zoom
        
        h, w = image.shape[1], image.shape[2]
        zoom_h = self.img_size / h
        zoom_w = self.img_size / w
        
        resized = zoom(image, (1, zoom_h, zoom_w), order=1)  # Bilinear interpolation
        return resized
    
    def _resize_mask(self, mask):
        """Resize segmentation mask using nearest neighbor."""
        from scipy.ndimage import zoom
        
        h, w = mask.shape
        zoom_h = self.img_size / h
        zoom_w = self.img_size / w
        
        resized = zoom(mask, (zoom_h, zoom_w), order=0)  # Nearest neighbor
        return resized
    
    def get_class_distribution(self):
        """Calculate class distribution across all masks."""
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for idx in range(len(self)):
            _, mask = self[idx]
            for c in range(4):
                class_counts[c] += (mask == c).sum().item()
        
        return class_counts


def create_segmentation_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    data_root="Dataset/brats_preprocessed/slices",
    batch_size=8,
    num_workers=4,
    augment_train=True,
    img_size=224
):
    """
    Create train, validation, and test dataloaders for segmentation.
    
    Args:
        train_csv: Path to training metadata CSV
        val_csv: Path to validation metadata CSV
        test_csv: Path to test metadata CSV
        data_root: Root directory for preprocessed slices
        batch_size: Batch size (usually smaller for segmentation due to memory)
        num_workers: Number of data loading workers
        augment_train: Whether to augment training data
        img_size: Target image size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Augmentation config for training
    augment_config = {
        'rotation': True,
        'flip': True,
        'intensity': True,
    } if augment_train else None
    
    # Create datasets
    train_dataset = BraTSSegmentationDataset(
        train_csv,
        data_root=data_root,
        augment=augment_train,
        augment_config=augment_config,
        img_size=img_size
    )
    
    val_dataset = BraTSSegmentationDataset(
        val_csv,
        data_root=data_root,
        augment=False,
        img_size=img_size
    )
    
    test_dataset = BraTSSegmentationDataset(
        test_csv,
        data_root=data_root,
        augment=False,
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
