"""
Evaluation Module: Test Models and Measure Distribution Shift

This module:
1. Evaluates models on CheXpert validation (in-distribution)
2. Evaluates models on ChestX-ray14 (out-of-distribution)
3. Computes performance metrics (AUC, sensitivity, specificity)
4. Measures distribution shift impact
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config


def evaluate_model(model, dataloader, device=None):
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained DenseNet model
        dataloader: Data to evaluate on
        device: CPU or CUDA
        
    Returns:
        results (dict): Contains predictions, labels, AUC scores, etc.
    """
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_paths = []
    
    print(f"Evaluating on {len(dataloader.dataset)} images...")
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            probs = torch.sigmoid(outputs)  # Convert to probabilities
            
            # Store results
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(probs.cpu().numpy())
            all_paths.extend(paths)
    
    # Concatenate batches
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Compute metrics
    auc_scores = {}
    for i, label_name in enumerate(config.CHEXPERT_LABELS):
        try:
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                auc_scores[label_name] = auc
            else:
                auc_scores[label_name] = None
        except:
            auc_scores[label_name] = None
    
    # Calculate average AUC
    valid_aucs = [auc for auc in auc_scores.values() if auc is not None]
    avg_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    
    results = {
        'labels': all_labels,
        'predictions': all_predictions,
        'paths': all_paths,
        'auc_scores': auc_scores,
        'avg_auc': avg_auc
    }
    
    return results


def compute_performance_metrics(labels, predictions, threshold=0.5):
    """
    Compute detailed metrics: sensitivity, specificity, etc.
    
    Args:
        labels (np.array): True labels [num_samples, num_classes]
        predictions (np.array): Predicted probabilities [num_samples, num_classes]
        threshold (float): Decision threshold (default 0.5)
        
    Returns:
        metrics (dict): Per-class metrics
    """
    metrics = {}
    
    for i, label_name in enumerate(config.CHEXPERT_LABELS):
        # Binary predictions
        y_true = labels[:, i]
        y_pred = (predictions[:, i] >= threshold).astype(int)
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            metrics[label_name] = None
            continue
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        
        metrics[label_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    return metrics


def compare_distributions(in_dist_results, out_dist_results):
    """
    Compare in-distribution vs out-of-distribution performance.
    
    Args:
        in_dist_results: Results from CheXpert validation
        out_dist_results: Results from ChestX-ray14
        
    Returns:
        comparison (pd.DataFrame): Table showing performance drop
    """
    comparison_data = []
    
    for label in config.CHEXPERT_LABELS:
        in_auc = in_dist_results['auc_scores'].get(label)
        out_auc = out_dist_results['auc_scores'].get(label)
        
        if in_auc is not None and out_auc is not None:
            drop = in_auc - out_auc
            drop_pct = (drop / in_auc * 100) if in_auc > 0 else 0
            
            comparison_data.append({
                'Disease': label,
                'CheXpert_AUC': in_auc,
                'ChestXray14_AUC': out_auc,
                'AUC_Drop': drop,
                'Drop_Percent': drop_pct
            })
    
    # Add average row
    avg_in = in_dist_results['avg_auc']
    avg_out = out_dist_results['avg_auc']
    avg_drop = avg_in - avg_out
    avg_drop_pct = (avg_drop / avg_in * 100) if avg_in > 0 else 0
    
    comparison_data.append({
        'Disease': 'AVERAGE',
        'CheXpert_AUC': avg_in,
        'ChestXray14_AUC': avg_out,
        'AUC_Drop': avg_drop,
        'Drop_Percent': avg_drop_pct
    })
    
    df = pd.DataFrame(comparison_data)
    return df


def plot_roc_curves(results, save_path=None):
    """
    Plot ROC curves for all diseases.
    
    Args:
        results: Evaluation results
        save_path (str, optional): Where to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for i, label in enumerate(config.CHEXPERT_LABELS):
        y_true = results['labels'][:, i]
        y_pred = results['predictions'][:, i]
        
        # Skip if only one class
        if len(np.unique(y_true)) < 2:
            continue
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = results['auc_scores'][label]
        
        # Plot
        plt.plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})')
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-Label Classification')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC plot to {save_path}")
    
    plt.show()


def create_comparison_table(strategy_results):
    """
    Create a comparison table across all three uncertainty strategies.
    
    Args:
        strategy_results (dict): Results for each strategy
            {
                'U_zeros': {'in_dist': ..., 'out_dist': ...},
                'U_ones': {'in_dist': ..., 'out_dist': ...},
                'U_half': {'in_dist': ..., 'out_dist': ...}
            }
    
    Returns:
        table (pd.DataFrame): Formatted comparison table
    """
    rows = []
    
    for strategy_name, results in strategy_results.items():
        in_dist = results['in_dist']
        out_dist = results['out_dist']
        
        # Calculate metrics
        in_auc = in_dist['avg_auc']
        out_auc = out_dist['avg_auc']
        drop = in_auc - out_auc
        drop_pct = (drop / in_auc * 100) if in_auc > 0 else 0
        
        rows.append({
            'Strategy': strategy_name,
            'CheXpert_Valid_AUC': f"{in_auc:.4f}",
            'ChestXray14_AUC': f"{out_auc:.4f}",
            'Performance_Drop': f"{drop:.4f}",
            'Drop_Percentage': f"{drop_pct:.2f}%"
        })
    
    table = pd.DataFrame(rows)
    
    # Sort by drop (ascending = better robustness)
    table = table.sort_values('Performance_Drop')
    
    return table


def visualize_drift_impact(comparison_df, save_path=None):
    """
    Create visualization showing drift impact across strategies.
    
    Args:
        comparison_df: DataFrame from create_comparison_table
        save_path (str, optional): Where to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    strategies = comparison_df['Strategy'].values
    in_dist_aucs = comparison_df['CheXpert_Valid_AUC'].astype(float).values
    out_dist_aucs = comparison_df['ChestXray14_AUC'].astype(float).values
    
    # Plot 1: Grouped bar chart
    x = np.arange(len(strategies))
    width = 0.35
    
    ax1.bar(x - width/2, in_dist_aucs, width, label='CheXpert (In-Dist)', alpha=0.8)
    ax1.bar(x + width/2, out_dist_aucs, width, label='ChestX-ray14 (OOD)', alpha=0.8)
    
    ax1.set_xlabel('Uncertainty Strategy')
    ax1.set_ylabel('Average AUC-ROC')
    ax1.set_title('In-Distribution vs Out-of-Distribution Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Performance drop
    drops = (in_dist_aucs - out_dist_aucs) * 100  # Convert to percentage points
    colors = ['green' if d == min(drops) else 'orange' for d in drops]
    
    ax2.bar(strategies, drops, color=colors, alpha=0.7)
    ax2.set_xlabel('Uncertainty Strategy')
    ax2.set_ylabel('Performance Drop (percentage points)')
    ax2.set_title('Distribution Shift Impact')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def print_results_summary(results, dataset_name="Dataset"):
    """
    Print a nice summary of evaluation results.
    
    Args:
        results: Evaluation results dictionary
        dataset_name: Name for display
    """
    print(f"\n{'='*80}")
    print(f"Evaluation Results: {dataset_name}")
    print(f"{'='*80}\n")
    
    print(f"Number of images: {len(results['labels'])}")
    print(f"Average AUC-ROC: {results['avg_auc']:.4f}\n")
    
    print("Per-Class AUC Scores:")
    print("-" * 40)
    for label, auc in results['auc_scores'].items():
        if auc is not None:
            print(f"  {label:20s}: {auc:.4f}")
        else:
            print(f"  {label:20s}: N/A")
    
    print(f"\n{'='*80}\n")
