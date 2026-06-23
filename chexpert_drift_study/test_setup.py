"""
Quick Test Script

Run this before starting full training to verify:
1. All packages are installed correctly
2. GPU is available
3. Data paths are configured
4. Basic data loading works

import os
import sys

def test_imports():
    """Test if all required packages are installed."""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
        
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
        
        import numpy as np
        print(f"✓ numpy {np.__version__}")
        
        from PIL import Image
        print(f"✓ Pillow (PIL)")
        
        from sklearn.metrics import roc_auc_score
        print(f"✓ scikit-learn")
        
        import matplotlib
        print(f"✓ matplotlib")
        
        import seaborn
        print(f"✓ seaborn")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠ No GPU detected - training will be VERY slow")
        print("  Consider using Google Colab or a GPU server")
        return False


def test_paths():
    """Test if data paths are configured."""
    print("\nTesting data paths...")
    
    import config
    
    paths_ok = True
    
    # Check CheXpert
    if os.path.exists(config.CHEXPERT_DIR):
        print(f"✓ CheXpert directory found: {config.CHEXPERT_DIR}")
        
        if os.path.exists(config.CHEXPERT_TRAIN_CSV):
            print(f"  ✓ train.csv found")
        else:
            print(f"  ✗ train.csv NOT found at {config.CHEXPERT_TRAIN_CSV}")
            paths_ok = False
            
        if os.path.exists(config.CHEXPERT_VALID_CSV):
            print(f"  ✓ valid.csv found")
        else:
            print(f"  ✗ valid.csv NOT found at {config.CHEXPERT_VALID_CSV}")
            paths_ok = False
    else:
        print(f"✗ CheXpert directory NOT found: {config.CHEXPERT_DIR}")
        print("  Please update CHEXPERT_DIR in config.py")
        paths_ok = False
    
    # Check ChestX-ray14
    if os.path.exists(config.CHESTXRAY14_DIR):
        print(f"✓ ChestX-ray14 directory found: {config.CHESTXRAY14_DIR}")
    else:
        print(f"⚠ ChestX-ray14 directory NOT found: {config.CHESTXRAY14_DIR}")
        print("  (Only needed for evaluation phase)")
    
    return paths_ok


def test_data_loading():
    """Test basic data loading."""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        import config
        
        # Try loading train CSV
        df = pd.read_csv(config.CHEXPERT_TRAIN_CSV)
        print(f"✓ Loaded training CSV: {len(df)} images")
        
        # Check if labels exist
        missing_labels = [l for l in config.CHEXPERT_LABELS if l not in df.columns]
        if missing_labels:
            print(f"✗ Missing labels: {missing_labels}")
            return False
        
        print(f"✓ All {len(config.CHEXPERT_LABELS)} labels present")
        
        # Check uncertainty distribution
        for label in config.CHEXPERT_LABELS[:2]:  # Just check first 2
            n_uncertain = (df[label] == -1.0).sum()
            if n_uncertain > 0:
                print(f"  {label}: {n_uncertain} uncertain labels")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from utils.model import create_model, count_parameters
        import config
        
        model = create_model()
        total, trainable = count_parameters(model)
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {trainable:,} trainable")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("CheXpert Drift Study - Setup Verification")
    print("="*80 + "\n")
    
    results = []
    
    results.append(("Package imports", test_imports()))
    results.append(("GPU availability", test_gpu()))
    results.append(("Data paths", test_paths()))
    results.append(("Data loading", test_data_loading()))
    results.append(("Model creation", test_model_creation()))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80 + "\n")
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to start training.\n")
        print("Next steps:")
        print("  1. Review config.py settings (batch size, epochs, etc.)")
        print("  2. Run: python main.py --mode train")
        print()
    else:
        print("\n⚠ Some tests failed. Please fix the issues above before training.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
