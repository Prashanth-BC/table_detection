#!/usr/bin/env python3
"""
Universal platform setup script for table detection training.
Automatically detects and configures for Google Colab, Kaggle, or local environments.
"""

import os
import sys
from pathlib import Path
from scripts.platform_detector import get_platform_config

def setup_platform():
    """Setup platform-specific environment for table detection training"""

    print("🔧 Table Detection Platform Setup")
    print("=" * 50)

    # Get platform configuration
    platform_config = get_platform_config()

    # Display platform information
    platform_config.print_platform_info()

    # Setup directories
    print("\n📁 Setting up directories...")
    platform_config.setup_directories()

    # Platform-specific optimizations
    print(f"\n⚙️ Applying {platform_config.platform.upper()} optimizations...")

    if platform_config.platform == 'kaggle':
        setup_kaggle_optimizations(platform_config)
    elif platform_config.platform == 'colab':
        setup_colab_optimizations(platform_config)
    else:
        setup_local_optimizations(platform_config)

    # Display training recommendations
    print_training_recommendations(platform_config)

    return platform_config

def setup_kaggle_optimizations(config):
    """Apply Kaggle-specific optimizations"""

    # Create additional symlinks for convenience
    symlinks = [
        ("/kaggle/working/data", "/kaggle/tmp/data"),
        ("/kaggle/working/models", "/kaggle/tmp/models")
    ]

    for link_path, target_path in symlinks:
        link = Path(link_path)
        if not link.exists() and Path(target_path).exists():
            link.symlink_to(target_path)
            print(f"✓ Created symlink: {link_path} -> {target_path}")

    # Set environment variables for optimal performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
    os.environ['TORCH_HOME'] = '/kaggle/tmp/.torch'  # Cache torch models

    print("✓ Kaggle optimizations applied:")
    print("  • Symlinks created for easy data access")
    print("  • CUDA debugging enabled")
    print("  • Torch cache redirected to /kaggle/tmp/")

def setup_colab_optimizations(config):
    """Apply Google Colab-specific optimizations"""

    # Check if GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ No GPU detected. Enable GPU: Runtime → Change runtime type → GPU")
    except ImportError:
        print("⚠️ PyTorch not installed")

    # Set environment variables
    os.environ['TORCH_HOME'] = '/content/.torch'  # Cache torch models

    print("✓ Colab optimizations applied:")
    print("  • Torch cache configured")
    print("  • GPU status checked")

def setup_local_optimizations(config):
    """Apply local environment optimizations"""

    print("✓ Local environment detected:")
    print("  • Using relative paths for data")
    print("  • Full hardware access available")
    print("  • No time or storage limitations")

def print_training_recommendations(config):
    """Print platform-specific training recommendations"""

    print(f"\n🎯 {config.platform.upper()} Training Recommendations:")
    print("=" * 50)

    if config.platform == 'kaggle':
        print("🔥 KAGGLE TIPS:")
        print("• Use 30-hour GPU quota efficiently")
        print("• Save checkpoints frequently (every 5 epochs)")
        print("• Upload large datasets as Kaggle datasets")
        print("• Use /kaggle/tmp/ for temporary large files")
        print("• Save final models in /kaggle/working/ for download")
        print("• Consider using Kaggle's built-in datasets")

    elif config.platform == 'colab':
        print("🔥 COLAB TIPS:")
        print("• 12-hour session limit - save work frequently")
        print("• Use Google Drive for persistent storage")
        print("• Enable GPU: Runtime → Change runtime type → GPU")
        print("• Download models before session expires")
        print("• Use smaller batch sizes if memory limited")
        print("• Mount Google Drive: files.upload() for data")

    else:
        print("🔥 LOCAL TIPS:")
        print("• Full control over hardware and time")
        print("• Use larger batch sizes if GPU memory allows")
        print("• Enable distributed training for multiple GPUs")
        print("• Use full datasets for best results")
        print("• Consider using Docker for reproducibility")

    print(f"\n📊 Quick Start Commands:")
    print(f"1. Setup data: python scripts/data_converter.py")
    print(f"2. Start training: python train_table_detection.py")
    print(f"3. Check results: ls {config.config['results_dir']}/")

def verify_setup(config):
    """Verify that setup was successful"""

    print(f"\n🔍 Verifying setup...")

    # Check critical directories
    critical_dirs = [
        config.config['data_root'],
        config.config['checkpoints_dir'],
        config.config['logs_dir'],
        config.config['results_dir']
    ]

    missing_dirs = []
    for dir_path in critical_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✓ {dir_path}")

    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False

    print("✅ All directories verified!")
    return True

def main():
    """Main setup function"""

    try:
        # Setup platform
        config = setup_platform()

        # Verify setup
        if verify_setup(config):
            print(f"\n🎉 {config.platform.upper()} setup completed successfully!")
            print("You're ready to start table detection training!")
        else:
            print(f"\n❌ Setup verification failed!")
            return False

        return True

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)