#!/usr/bin/env python3
"""
Quick setup script specifically optimized for Kaggle environments.
Handles Kaggle-specific paths, symlinks, and optimizations.
"""

import os
import sys
from pathlib import Path
import shutil

def kaggle_quickstart():
    """Quick setup for Kaggle environment"""

    print("🟠 KAGGLE Table Detection Quick Setup")
    print("=" * 50)

    # Verify we're on Kaggle
    if not os.path.exists('/kaggle'):
        print("❌ This script is designed for Kaggle environments!")
        print("Use setup_platform.py for automatic platform detection.")
        return False

    print("✓ Kaggle environment detected")

    # Create directory structure
    kaggle_dirs = [
        # Temporary storage (73GB available)
        "/kaggle/tmp/data/raw",
        "/kaggle/tmp/data/processed",
        "/kaggle/tmp/data/yolo/train/images",
        "/kaggle/tmp/data/yolo/train/labels",
        "/kaggle/tmp/data/yolo/val/images",
        "/kaggle/tmp/data/yolo/val/labels",
        "/kaggle/tmp/data/yolo/test/images",
        "/kaggle/tmp/data/yolo/test/labels",

        # Working directory (persistent, 20GB)
        "/kaggle/working/checkpoints",
        "/kaggle/working/logs",
        "/kaggle/working/results",
        "/kaggle/working/final_models",
        "/kaggle/working/configs"
    ]

    print("📁 Creating Kaggle directory structure...")
    for dir_path in kaggle_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")

    # Create convenient symlinks
    symlinks = [
        ("/kaggle/working/data", "/kaggle/tmp/data"),
        ("/kaggle/working/tmp", "/kaggle/tmp")
    ]

    print("\n🔗 Creating convenience symlinks...")
    for link_path, target_path in symlinks:
        link = Path(link_path)
        if not link.exists():
            link.symlink_to(target_path)
            print(f"✓ {link_path} -> {target_path}")

    # Copy essential configs to working directory
    if os.path.exists("configs"):
        print("\n📋 Copying configs to working directory...")
        shutil.copytree("configs", "/kaggle/working/configs", dirs_exist_ok=True)
        print("✓ Configs copied to /kaggle/working/configs/")

    # Set Kaggle-specific environment variables
    print("\n⚙️ Setting Kaggle optimizations...")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_HOME'] = '/kaggle/tmp/.torch'
    os.environ['TRANSFORMERS_CACHE'] = '/kaggle/tmp/.cache'
    print("✓ Environment variables set")

    # Create dataset.yaml template
    create_kaggle_dataset_yaml()

    # Display usage information
    print_kaggle_usage_info()

    return True

def create_kaggle_dataset_yaml():
    """Create YOLO dataset configuration for Kaggle"""

    dataset_yaml = """# YOLO dataset configuration for Kaggle
path: /kaggle/tmp/data/yolo  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')
test: test/images    # test images (optional, relative to 'path')

# Classes
names:
  0: table
  1: table_column
  2: table_row
  3: table_column_header
  4: table_projected_row_header
  5: table_spanning_cell

# Number of classes
nc: 6
"""

    yaml_path = "/kaggle/tmp/data/yolo/dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)

    print(f"✓ Created dataset.yaml at {yaml_path}")

def print_kaggle_usage_info():
    """Print Kaggle-specific usage information"""

    print(f"\n🎯 KAGGLE QUICK START COMMANDS:")
    print("=" * 50)
    print("# 1. Install dependencies")
    print("!pip install ultralytics pyyaml pillow tqdm")
    print()
    print("# 2. Create sample dataset")
    print("!python create_sample_dataset.py --output_dir /kaggle/tmp/data/raw/sample_tables")
    print()
    print("# 3. Convert to YOLO format")
    print("!python scripts/data_converter.py configs/data_config.yaml")
    print()
    print("# 4. Start training")
    print("!python train_table_detection.py")
    print()

    print(f"\n🗂️ KAGGLE DIRECTORY STRUCTURE:")
    print("=" * 30)
    print("/kaggle/tmp/data/          - Large training data (73GB)")
    print("/kaggle/working/           - Persistent files (20GB)")
    print("/kaggle/input/             - Input datasets")
    print("/kaggle/output/            - Notebook output")
    print()

    print(f"\n💡 KAGGLE TIPS:")
    print("=" * 15)
    print("• 30-hour GPU quota per week")
    print("• 9-hour maximum session length")
    print("• Save important files in /kaggle/working/")
    print("• Use /kaggle/tmp/ for large temporary files")
    print("• Upload datasets as Kaggle datasets for reuse")
    print("• Enable GPU: Settings → Accelerator → GPU")
    print("• Download models before session expires")
    print()

    print(f"⚡ PERFORMANCE OPTIMIZATION:")
    print("=" * 30)
    print("• Use batch_size=16 for optimal GPU utilization")
    print("• Set workers=2 for data loading")
    print("• Enable mixed precision: amp=True")
    print("• Save checkpoints every 5 epochs")
    print("• Monitor GPU usage: !nvidia-smi")

def verify_kaggle_setup():
    """Verify Kaggle setup is working correctly"""

    print(f"\n🔍 Verifying Kaggle setup...")

    # Check directories
    required_dirs = [
        "/kaggle/tmp/data",
        "/kaggle/working/checkpoints",
        "/kaggle/working/logs",
        "/kaggle/working/results"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            missing_dirs.append(dir_path)
            print(f"❌ {dir_path}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
        else:
            print("⚠️ No GPU detected - enable in Settings → Accelerator → GPU")
    except ImportError:
        print("⚠️ PyTorch not installed")

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/kaggle/tmp")
    tmp_gb = free // (1024**3)

    total, used, free = shutil.disk_usage("/kaggle/working")
    working_gb = free // (1024**3)

    print(f"💾 Available storage: /kaggle/tmp: {tmp_gb}GB, /kaggle/working: {working_gb}GB")

    if missing_dirs:
        print(f"\n❌ Setup incomplete - missing directories: {missing_dirs}")
        return False
    else:
        print(f"\n✅ Kaggle setup verified successfully!")
        return True

def main():
    """Main setup function"""

    try:
        # Run setup
        success = kaggle_quickstart()

        if success:
            # Verify setup
            if verify_kaggle_setup():
                print(f"\n🎉 Kaggle environment ready for table detection training!")
                return True
            else:
                print(f"\n⚠️ Setup completed but verification failed")
                return False
        else:
            print(f"\n❌ Kaggle setup failed!")
            return False

    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)