"""
Custom Dataset Classes for CheXpert and ChestX-ray14

This module handles:
1. Loading images and labels from CSV files
2. Applying uncertainty strategies to CheXpert labels
3. Data augmentation and normalization
4. Label mapping between datasets
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config

class CheXpertDataset(Dataset):
    """
    CheXpert Dataset with configurable uncertainty handling.
    
    Args:
        csv_file (str): Path to train.csv or valid.csv
        uncertainty_strategy (float): How to handle uncertain labels
            - 0.0: Uncertain -> Negative (U-Zeros)
            - 1.0: Uncertain -> Positive (U-Ones)
            - 0.5: Uncertain -> Soft label (U-Half)
        transform (callable, optional): Image transformations
        labels (list): Which pathologies to include
    """
    
    def __init__(self, csv_file, uncertainty_strategy=0.0, transform=None, labels=None):
        self.data_dir = config.CHEXPERT_DIR
        self.df = pd.read_csv(csv_file)
        self.uncertainty_strategy = uncertainty_strategy
        self.transform = transform
        self.labels = labels if labels else config.CHEXPERT_LABELS
        
        # Clean the dataframe
        self._prepare_labels()
        
        print(f"Loaded {len(self.df)} images from {csv_file}")
        print(f"Uncertainty strategy: {uncertainty_strategy}")
        print(f"Labels: {self.labels}")
    
    def _prepare_labels(self):
        """
        Handle missing and uncertain labels.
        
        CheXpert label encoding:
        - Blank (NaN): Not mentioned -> treat as 0 (negative)
        - 0.0: Negative (explicitly stated)
        - -1.0: Uncertain (radiologist unsure)
        - 1.0: Positive (explicitly stated)
        """
        # Fill NaN with 0.0 (not mentioned = negative)
        self.df[self.labels] = self.df[self.labels].fillna(0.0)
        
        # Replace uncertain (-1.0) with the strategy
        for label in self.labels:
            # Find uncertain entries
            uncertain_mask = self.df[label] == -1.0
            
            # Replace with strategy value
            self.df.loc[uncertain_mask, label] = self.uncertainty_strategy
            
            # Count how many uncertains 
            n_uncertain = uncertain_mask.sum()
            if n_uncertain > 0:
                print(f"  {label}: {n_uncertain} uncertain labels → {self.uncertainty_strategy}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get one sample (image + labels).
        
        Returns:
            image (torch.Tensor): Preprocessed image [3, 224, 224]
            labels (torch.Tensor): Multi hot label vector [num_classes]
            path (str): Image path 
        """
        # Get image path
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['Path'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get labels for this image
        labels = self.df.iloc[idx][self.labels].values.astype('float32')
        labels = torch.tensor(labels)
        
        # Apply transformations (augmentation + normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, labels, img_path


class ChestXray14Dataset(Dataset):
    """
    ChestX-ray14 Dataset for out of distribution testing.
    
    Args:
        csv_file (str): Path to Data_Entry_2017.csv or test_list.txt
        image_dir (str): Path to images folder
        transform (callable, optional): Image transformations
        labels (list): Which pathologies to include (mapped to CheXpert)
    """
    
    def __init__(self, csv_file, image_dir, transform=None, labels=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels if labels else config.CHEXPERT_LABELS
        
        # Load data
        # ChestX-ray14 format: Image Index | Finding Labels | Follow-up # | ...
        self.df = pd.read_csv(csv_file)
        
        # Parse labels
        self._prepare_labels()
        
        print(f"Loaded {len(self.df)} images from ChestX-ray14")
        print(f"Mapped labels: {self.labels}")
    
    def _prepare_labels(self):
        """
        Parse ChestX-ray14 labels and map to CheXpert format.
        
        ChestX-ray14 stores labels as pipe-separated strings:
        "Cardiomegaly|Edema|Effusion"
        
        We need to convert this to multi hot encoding.
        """
        # Create empty label columns
        for label in config.CHEXPERT_LABELS:
            self.df[label] = 0.0
        
        # Parse each image's findings
        for idx, row in self.df.iterrows():
            findings = str(row['Finding Labels']).split('|')
            
            # Map ChestX-ray14 labels to CheXpert labels
            for finding in findings:
                finding = finding.strip()
                
                # Use mapping dictionary
                if finding in config.CHESTXRAY14_TO_CHEXPERT_MAPPING:
                    chexpert_label = config.CHESTXRAY14_TO_CHEXPERT_MAPPING[finding]
                    if chexpert_label in self.labels:
                        self.df.loc[idx, chexpert_label] = 1.0
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get one sample."""
        # Get image filename
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get labels
        labels = self.df.iloc[idx][self.labels].values.astype('float32')
        labels = torch.tensor(labels)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, labels, img_path


def get_transforms(is_training=True):
    """
    Get image transformation pipeline.
    
    For training: Apply data augmentation
    For validation/testing: Only resize and normalize
    
    Args:
        is_training (bool): Whether this is for training data
        
    Returns:
        transforms.Compose: Transformation pipeline
    """
    if is_training:
        # Training: Apply augmentation to prevent overfitting
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),  # Flip left/right
            transforms.RandomRotation(10),            # Slight rotation
            transforms.ColorJitter(                   # Vary brightness/contrast
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    else:
        # Validation/Testing: No augmentation, just preprocess
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])


# HELPER FUNCTION: Create data loaders
def create_dataloaders(uncertainty_strategy):
    """
    Create train and validation dataloaders for a given uncertainty strategy.
    
    Args:
        uncertainty_strategy (float): 0.0, 0.5, or 1.0
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CheXpertDataset(
        csv_file=config.CHEXPERT_TRAIN_CSV,
        uncertainty_strategy=uncertainty_strategy,
        transform=get_transforms(is_training=True),
        labels=config.CHEXPERT_LABELS
    )
    
    val_dataset = CheXpertDataset(
        csv_file=config.CHEXPERT_VALID_CSV,
        uncertainty_strategy=uncertainty_strategy,  # Note: val set has few uncertains
        transform=get_transforms(is_training=False),
        labels=config.CHEXPERT_LABELS
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle training data
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # Don't shuffle validation
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader
