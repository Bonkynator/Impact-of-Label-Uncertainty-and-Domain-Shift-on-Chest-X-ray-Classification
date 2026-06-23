"""
Training Pipeline with Validation and Checkpointing

This module handles:
1. Training loop with backpropagation
2. Validation after each epoch
3. Early stopping based on validation performance
4. Model checkpointing (save best model)
5. Metrics tracking (loss, AUC)
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import config


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Stops training if validation loss doesn't improve for `patience` epochs.
    """
    
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait before stopping
            min_delta (float): Minimum improvement to count as progress
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Check if we should stop.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: Whether to stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # No improvement
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement!
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: The neural network
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer (Adam, SGD, etc.)
        device: CPU or CUDA
        epoch: Current epoch number
        
    Returns:
        avg_loss (float): Average training loss for this epoch
    """
    model.train()  # Set model to training mode 
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (images, labels, _) in enumerate(dataloader):
        # Move data to device (GPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients 
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(images)  # [batch_size, num_classes]
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f}")
    
    # Return average loss for this epoch
    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Validate the model (no gradient updates).
    
    Args:
        model: The neural network
        dataloader: Validation data loader
        criterion: Loss function
        device: CPU or CUDA
        
    Returns:
        avg_loss (float): Average validation loss
        auc_scores (dict): Per-class AUC-ROC scores
        avg_auc (float): Mean AUC across all classes
    """
    model.eval()  # Set model to evaluation mode 
    
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    # Disable gradient computation (saves memory and computation)
    with torch.no_grad():
        for images, labels, _ in dataloader:
            # Move to device
            images = images.to(device)
            labels_gpu = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels_gpu)
            running_loss += loss.item()
            
            # Get probabilities
            probs = torch.sigmoid(outputs)
            
            # Store for metric calculation
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(probs.cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels, axis=0)  # [num_samples, num_classes]
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Compute metrics
    avg_loss = running_loss / len(dataloader)
    
    # Compute AUC-ROC for each disease
    auc_scores = {}
    valid_aucs = []
    
    for i, label_name in enumerate(config.CHEXPERT_LABELS):
        try:
            # Check if we have both positive and negative examples
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                auc_scores[label_name] = auc
                valid_aucs.append(auc)
            else:
                auc_scores[label_name] = None  # Cannot compute AUC
        except Exception as e:
            print(f"Warning: Could not compute AUC for {label_name}: {e}")
            auc_scores[label_name] = None
    
    # Average AUC
    avg_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    
    return avg_loss, auc_scores, avg_auc


def train_model(model, train_loader, val_loader, uncertainty_strategy, num_epochs=None):
    """
    Complete training pipeline.
    
    This is the main function you'll call to train a model!
    
    Args:
        model: The DenseNet model
        train_loader: Training data
        val_loader: Validation data
        uncertainty_strategy (str): Name of strategy ('U_zeros', 'U_ones', 'U_half')
        num_epochs (int, optional): Number of epochs (uses config if None)
        
    Returns:
        model: Trained model
        history (dict): Training history (losses, AUCs per epoch)
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    
    # Create optimizer (Adam is standard for fine-tuning)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create loss function
    from utils.model import MultiLabelLoss
    criterion = MultiLabelLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA
    )
    
    # Track training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_auc_per_class': []
    }
    
    # Best model tracking
    best_auc = 0.0
    best_epoch = 0
    
    print(f"\n{'='*80}")
    print(f"Starting training: {uncertainty_strategy}")
    print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch
        )
        
        # Validate
        val_loss, auc_scores, avg_auc = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(avg_auc)
        history['val_auc_per_class'].append(auc_scores)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val AUC:    {avg_auc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Print per-class AUCs
        print(f"  Per-class AUCs:")
        for label, auc in auc_scores.items():
            if auc is not None:
                print(f"    {label:20s}: {auc:.4f}")
        
        # Save best model
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_epoch = epoch
            
            if config.SAVE_BEST_ONLY:
                model_path = os.path.join(
                    config.MODELS_DIR, 
                    f"best_model_{uncertainty_strategy}.pth"
                )
                from utils.model import save_model
                save_model(
                    model, model_path, epoch, optimizer,
                    metrics={'val_auc': avg_auc, 'val_loss': val_loss}
                )
                print(f"  ✓ Saved new best model (AUC: {avg_auc:.4f})")
        
        # Check early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        print(f"{'-'*80}\n")
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation AUC: {best_auc:.4f} (epoch {best_epoch})")
    print(f"{'='*80}\n")
    
    return model, history
