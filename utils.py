"""
Utility script for quick testing and setup verification.
"""

import torch
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *


def check_environment():
    """Check if the environment is properly set up."""
    print("Environment Check")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Check device configuration
    print(f"Configured device: {DEVICE}")
    
    # Check data path
    print(f"Data path: {DATA_PATH}")
    print(f"Data path exists: {os.path.exists(DATA_PATH)}")
    
    # Check directories
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Generated samples dir: {GENERATED_SAMPLES_DIR}")
    
    print("\nConfiguration Summary:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  SAR channels: {SAR_CHANNELS}")
    print(f"  EO channels: {EO_CHANNELS}")


def test_imports():
    """Test if all required modules can be imported."""
    print("\nImport Test")
    print("=" * 50)
    
    try:
        from preprocess import SARToEODataset, create_dataloaders
        print("✓ Preprocessing module imported successfully")
    except ImportError as e:
        print(f"✗ Preprocessing import failed: {e}")
    
    try:
        from models import create_models
        print("✓ Models module imported successfully")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
    
    try:
        from losses import CycleGANLosses
        print("✓ Losses module imported successfully")
    except ImportError as e:
        print(f"✗ Losses import failed: {e}")
    
    try:
        import train_cycleGAN
        print("✓ Training module imported successfully")
    except ImportError as e:
        print(f"✗ Training import failed: {e}")
    
    try:
        import evaluate_results
        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"✗ Evaluation import failed: {e}")


def test_model_creation():
    """Test model creation."""
    print("\nModel Creation Test")
    print("=" * 50)
    
    try:
        from models import create_models
        models = create_models()
        print("✓ Models created successfully")
        
        # Test model sizes
        for i, (name, model) in enumerate([
            ("SAR to EO Generator", models[0]),
            ("EO to SAR Generator", models[1]),
            ("SAR Critic", models[2]),
            ("EO Critic", models[3])
        ]):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {total_params:,} parameters")
            
    except Exception as e:
        print(f"✗ Model creation failed: {e}")


def main():
    """Run all tests."""
    print("SAR to EO CycleGAN - Setup Verification")
    print("=" * 80)
    
    check_environment()
    test_imports()
    test_model_creation()
    
    print("\nSetup verification completed!")
    print("If all tests pass, you're ready to start training.")


if __name__ == "__main__":
    main()
