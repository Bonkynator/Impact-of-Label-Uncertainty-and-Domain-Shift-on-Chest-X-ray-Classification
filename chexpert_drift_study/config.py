"""
Configuration file for CheXpert Data Drift Study

This file contains all hyperparameters, paths, and settings.
Modify these values to customize your experiments.
"""

import torch

# PATHS - Update these to match your data locations
CHEXPERT_DIR = ""  # <- Update
CHESTXRAY14_DIR = ""     # <- Update
RESULTS_DIR = "/results"
MODELS_DIR = "/models"

# CSV files for CheXpert
CHEXPERT_TRAIN_CSV = f"{CHEXPERT_DIR}/train.csv"
CHEXPERT_VALID_CSV = f"{CHEXPERT_DIR}/valid.csv"

# PATHOLOGIES - The diseases we'll predict
# CheXpert has 14 labels, we'll use the 5 most common competition labels
CHEXPERT_LABELS = [
    'Atelectasis',
    'Cardiomegaly', 
    'Consolidation',
    'Edema',
    'Pleural Effusion'
]

# ChestX-ray14 label mapping (they use different names)
# We'll map their labels to match CheXpert 
CHESTXRAY14_TO_CHEXPERT_MAPPING = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Effusion': 'Pleural Effusion' 
}

# UNCERTAINTY STRATEGIES 
# How to handle uncertain labels (-1.0 in CheXpert)
UNCERTAINTY_STRATEGIES = {
    'U_zeros': 0.0,    # Uncertain -> Negative (conservative)
    'U_ones': 1.0,     # Uncertain -> Positive (aggressive)  
    'U_half': 0.5      # Uncertain -> Soft label (middle ground)
}


# MODEL HYPERPARAMETERS
MODEL_NAME = 'densenet121'  # Using DenseNet-121 as in your plan
PRETRAINED = True           # Use ImageNet pretrained weights
NUM_CLASSES = len(CHEXPERT_LABELS)

# Training hyperparameters
BATCH_SIZE = 32            # based on your GPU memory
NUM_EPOCHS = 10            # Start with 10, increase to 20-30 for final runs
LEARNING_RATE = 1e-4       # Standard for fine-tuning
WEIGHT_DECAY = 1e-4        # L2 regularization

# Image preprocessing
IMG_SIZE = 224             # DenseNet standard input size
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]


# TRAINING SETTINGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4            # For data loading (adjust based on CPU cores)
PIN_MEMORY = True          # Faster data transfer to GPU

# Early stopping
PATIENCE = 5               # Stop if no improvement for 5 epochs
MIN_DELTA = 0.001          # Minimum improvement to count as progress


# REPRODUCIBILITY
RANDOM_SEED = 42           # For reproducible results


# LOGGING & SAVING
SAVE_BEST_ONLY = True      # Only save model if it improves
LOG_INTERVAL = 100         # Print training stats every N batches
SAVE_PREDICTIONS = True    # Save predictions for later analysis
