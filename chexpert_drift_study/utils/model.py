"""
Model Architecture: DenseNet-121 for Multi-Label Classification

This module:
1. Loads pretrained DenseNet from torchvision
2. Modifies the final layer for multi-label prediction
3. Handles model initialization and weight loading
"""

import torch
import torch.nn as nn
from torchvision import models
import config


class DenseNetClassifier(nn.Module):
    """
    DenseNet-121 adapted for multi-label chest X-ray classification.
    
    Key differences from standard classification:
    - Output layer has NUM_CLASSES outputs (not 1000 ImageNet classes)
    - Uses sigmoid activation (not softmax) for multi-label
    - Each disease is predicted independently
    
    Args:
        num_classes (int): Number of diseases to predict
        pretrained (bool): Whether to use ImageNet pretrained weights
    """
    
    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED):
        super(DenseNetClassifier, self).__init__()
        
        # Load DenseNet-121 from torchvision
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Get number of features in the last layer
        num_features = self.densenet.classifier.in_features
        
        # Replace the classifier layer
        # Original: Linear(1024 -> 1000) for ImageNet
        # New: Linear(1024 -> num_classes) for our diseases
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
        print(f"Created DenseNet-121 with {num_classes} outputs")
        print(f"Pretrained: {pretrained}")
        print(f"Feature dimension: {num_features}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 224, 224]
            
        Returns:
            logits (torch.Tensor): Raw predictions [batch_size, num_classes]
                (apply sigmoid to get probabilities)
        """
        return self.densenet(x)
    
    def predict_proba(self, x):
        """
        Get probability predictions (applies sigmoid).
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            probs (torch.Tensor): Probabilities [batch_size, num_classes]
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)  # Convert logits to [0, 1]
        return probs


def create_model():
    """
    Factory function to create a new model instance.
    
    Returns:
        model (DenseNetClassifier): Initialized model
    """
    model = DenseNetClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    )
    
    # Move to GPU if available
    model = model.to(config.DEVICE)
    
    return model


def count_parameters(model):
    """
    Count trainable parameters in the model.
    
    Useful for reporting in your paper's methods section.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def save_model(model, filepath, epoch, optimizer=None, metrics=None):
    """
    Save model checkpoint with training state.
    
    Args:
        model: The model to save
        filepath: Where to save it
        epoch: Current epoch number
        optimizer: Optimizer state (optional)
        metrics: Dict of metrics to save (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': config.NUM_CLASSES,
            'pretrained': config.PRETRAINED
        }
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_model(filepath, device=None):
    """
    Load a saved model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict (contains epoch, metrics, etc.)
    """
    if device is None:
        device = config.DEVICE
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model with saved config
    model = DenseNetClassifier(
        num_classes=checkpoint['model_config']['num_classes'],
        pretrained=False  # Don't reload ImageNet weights
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded model from {filepath}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, checkpoint



# LOSS FUNCTION: Binary Cross-Entropy for Multi-Label
class MultiLabelLoss(nn.Module):
    """
    Binary Cross-Entropy loss for multi-label classification.
    
    Why BCE and not CrossEntropy?
    - CrossEntropy: For single-label (mutually exclusive classes)
    - BCE: For multi-label (patient can have multiple diseases)
    
    Each disease is predicted independently with its own binary classifier.
    """
    
    def __init__(self, pos_weight=None):
        """
        Args:
            pos_weight (torch.Tensor, optional): Weight for positive class
                Useful if you have class imbalance (more negatives than positives)
        """
        super(MultiLabelLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, logits, targets):
        """
        Compute loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs [batch_size, num_classes]
            targets (torch.Tensor): Ground truth labels [batch_size, num_classes]
                Can be 0, 1, or 0.5 (soft labels from U-Half strategy!)
        
        Returns:
            loss (torch.Tensor): Scalar loss value
        """
        # BCEWithLogitsLoss applies sigmoid internally, so pass raw logits
        return self.criterion(logits, targets)


def create_loss_function(pos_weights=None):
    """
    Factory function to create loss function.
    
    Args:
        pos_weights (list, optional): Per-class positive weights for imbalance
        
    Returns:
        criterion: Loss function
    """
    if pos_weights is not None:
        pos_weights = torch.tensor(pos_weights).to(config.DEVICE)
    
    return MultiLabelLoss(pos_weight=pos_weights)
