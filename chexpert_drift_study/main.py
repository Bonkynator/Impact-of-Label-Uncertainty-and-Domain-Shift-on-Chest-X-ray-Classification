"""
Main Script: Train and Evaluate All Models

Usage:
    python main.py --mode train    # Train all 3 models
    python main.py --mode eval     # Evaluate all 3 models
    python main.py --mode all      # Train + evaluate everything
"""

import os
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import our custom modules
import config
from utils.dataset import create_dataloaders, ChestXray14Dataset, get_transforms
from utils.model import create_model, count_parameters, load_model
from utils.train import train_model
from utils.evaluate import (
    evaluate_model, compare_distributions, create_comparison_table,
    visualize_drift_impact, print_results_summary
)


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    print(" Directories created")


def train_all_strategies():
    """
    Train models for all three uncertainty strategies.
    
    This will train 3 separate models:
    1. U-Zeros: Uncertain → 0
    2. U-Ones: Uncertain → 1
    3. U-Half: Uncertain → 0.5
    """
    print("\n" + "-"*80)
    print("TRAINING PHASE: All Uncertainty Strategies")
    print("-"*80 + "\n")
    
    results_summary = {}
    
    for strategy_name, strategy_value in config.UNCERTAINTY_STRATEGIES.items():
        print(f"\n{'#'*80}")
        print(f"# Training Strategy: {strategy_name} (uncertain → {strategy_value})")
        print(f"{'#'*80}\n")
        
        # Create data loaders for this strategy
        train_loader, val_loader = create_dataloaders(strategy_value)
        
        # Create model
        model = create_model()
        count_parameters(model)
        
        # Train model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            uncertainty_strategy=strategy_name,
            num_epochs=config.NUM_EPOCHS
        )
        
        # Save training history
        history_path = os.path.join(config.RESULTS_DIR, f"history_{strategy_name}.json")
        with open(history_path, 'w') as f:
            # convert numpy values to Python types for JSON
            history_json = {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_auc': [float(x) for x in history['val_auc']]
            }
            json.dump(history_json, f, indent=2)
        
        print(f"Saved training history to {history_path}")
        
        results_summary[strategy_name] = {
            'final_val_auc': history['val_auc'][-1],
            'best_val_auc': max(history['val_auc'])
        }
    
    # Print summary
    print("\n" + "-"*80)
    print("TRAINING SUMMARY")
    print("-"*80 + "\n")
    
    for strategy, metrics in results_summary.items():
        print(f"{strategy}:")
        print(f"  ->Best Val AUC: {metrics['best_val_auc']:.4f}")
        print(f"  ->Final Val AUC: {metrics['final_val_auc']:.4f}\n")
    
    print("---All models trained successfully---\n")


def evaluate_all_strategies():
    """
    Evaluate all trained models on both datasets.
    
    For each model:
    1. Load the trained weights
    2. Evaluate on CheXpert validation (in-distribution)
    3. Evaluate on ChestX-ray14 (out-of-distribution)
    4. Compute drift metrics
    """
    print("\n" + "-"*80)
    print("EVALUATION PHASE: Testing Distribution Shift")
    print("-"*80 + "\n")
    
    # Load ChestX-ray14 test set
    print("---Loading ChestX-ray14 dataset--")
    chestxray14_dataset = ChestXray14Dataset(
        csv_file=f"{config.CHESTXRAY14_DIR}/Data_Entry_2017.csv",
        image_dir=f"{config.CHESTXRAY14_DIR}/images-224",
        transform=get_transforms(is_training=False),
        labels=config.CHEXPERT_LABELS
    )
    
    chestxray14_loader = DataLoader(
        chestxray14_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Store results for all strategies
    all_results = {}
    
    for strategy_name in config.UNCERTAINTY_STRATEGIES.keys():
        print(f"\n{'-'*80}")
        print(f"Evaluating Strategy: {strategy_name}")
        print(f"{'-'*80}\n")
        
        # Load trained model
        model_path = os.path.join(config.MODELS_DIR, f"best_model_{strategy_name}.pth")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Run with --mode train first")
            continue
        
        model, checkpoint = load_model(model_path, device=config.DEVICE)
        
        # Create CheXpert validation loader (for in distribution testing)
        _, val_loader = create_dataloaders(
            config.UNCERTAINTY_STRATEGIES[strategy_name]
        )
        
        # Evaluate on CheXpert validation (in distribution)
        print("\n--- In-Distribution Evaluation (CheXpert Validation) ---")
        in_dist_results = evaluate_model(model, val_loader, config.DEVICE)
        print_results_summary(in_dist_results, "CheXpert Validation")
        
        # Evaluate on ChestX-ray14 (out of distribution)
        print("\n--- Out-of-Distribution Evaluation (ChestX-ray14) ---")
        out_dist_results = evaluate_model(model, chestxray14_loader, config.DEVICE)
        print_results_summary(out_dist_results, "ChestX-ray14")
        
        # Compute distribution shift
        print("\n--- Distribution Shift Analysis ---")
        comparison = compare_distributions(in_dist_results, out_dist_results)
        print(comparison.to_string(index=False))
        
        # Save results
        all_results[strategy_name] = {
            'in_dist': in_dist_results,
            'out_dist': out_dist_results,
            'comparison': comparison
        }
        
        # Save comparison to CSV
        comparison_path = os.path.join(
            config.RESULTS_DIR, f"drift_analysis_{strategy_name}.csv"
        )
        comparison.to_csv(comparison_path, index=False)
        print(f"\n✓ Saved drift analysis to {comparison_path}")
    
    # Create final comparison table across all strategies
    print("\n" + "-"*80)
    print("FINAL COMPARISON: All Strategies")
    print("-"*80 + "\n")
    
    final_table = create_comparison_table(all_results)
    print(final_table.to_string(index=False))
    
    # Save final table
    final_table_path = os.path.join(config.RESULTS_DIR, "final_comparison.csv")
    final_table.to_csv(final_table_path, index=False)
    print(f"\n Saved final comparison to {final_table_path}")
    
    # Create visualization
    viz_path = os.path.join(config.RESULTS_DIR, "drift_comparison.png")
    visualize_drift_impact(final_table, save_path=viz_path)
    
    # Print conclusion
    print("\n" + "-"*80)
    print("KEY FINDINGS")
    print("-"*80 + "\n")
    
    # Find most robust strategy 
    final_table['Drop_Value'] = final_table['Performance_Drop'].astype(float)
    best_strategy = final_table.loc[final_table['Drop_Value'].idxmin(), 'Strategy']
    
    print(f"Most robust to distribution shift: {best_strategy}")
    print("\nThis is your main result for the paper!")
    print("-"*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CheXpert Data Drift Study - Train and evaluate models'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'all'],
        default='all',
        help='What to run: train models, evaluate models, or both'
    )
    
    args = parser.parse_args()
    
    # Setup
    print("\n" + "-"*80)
    print("CheXpert Data Drift Study")
    print("Uncertainty Handling and Distribution Shift Analysis")
    print("-"*80 + "\n")
    
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}\n")
    
    setup_directories()
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Execute requested mode
    if args.mode in ['train', 'all']:
        train_all_strategies()
    
    if args.mode in ['eval', 'all']:
        evaluate_all_strategies()
    
    print("\n" + "-"*80)
    print("Pipeline completed successfully")
    print("-"*80 + "\n")


if __name__ == "__main__":
    main()
