#!/usr/bin/env python3
"""
Quick setup script specifically optimized for Google Colab environments.
Handles Colab-specific paths, GPU setup, and Google Drive integration.
"""

import os
import sys
from pathlib import Path
import shutil

def colab_quickstart():
    """Quick setup for Google Colab environment"""

    print("🔵 GOOGLE COLAB Table Detection Quick Setup")
    print("=" * 50)

    # Verify we're on Colab
    if not os.path.exists('/content'):
        print("❌ This script is designed for Google Colab environments!")
        print("Use setup_platform.py for automatic platform detection.")
        return False

    print("✓ Google Colab environment detected")

    # Check GPU availability
    check_gpu_setup()

    # Create directory structure
    colab_dirs = [
        "/content/data/raw",
        "/content/data/processed",
        "/content/data/yolo/train/images",
        "/content/data/yolo/train/labels",
        "/content/data/yolo/val/images",
        "/content/data/yolo/val/labels",
        "/content/data/yolo/test/images",
        "/content/data/yolo/test/labels",
        "/content/checkpoints",
        "/content/logs",
        "/content/results",
        "/content/models"
    ]

    print("📁 Creating Colab directory structure...")
    for dir_path in colab_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")

    # Set Colab-specific environment variables
    print("\n⚙️ Setting Colab optimizations...")
    os.environ['TORCH_HOME'] = '/content/.torch'
    os.environ['TRANSFORMERS_CACHE'] = '/content/.cache'
    print("✓ Environment variables set")

    # Create dataset.yaml template
    create_colab_dataset_yaml()

    # Offer Google Drive integration
    setup_google_drive_integration()

    # Display usage information
    print_colab_usage_info()

    return True

def check_gpu_setup():
    """Check and display GPU information"""

    print("\n🔍 Checking GPU setup...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ No GPU detected!")
            print("💡 To enable GPU:")
            print("   Runtime → Change runtime type → Hardware accelerator → GPU")
            print("   Then restart the runtime")
    except ImportError:
        print("⚠️ PyTorch not installed - will install during dependency setup")

def create_colab_dataset_yaml():
    """Create YOLO dataset configuration for Colab"""

    dataset_yaml = """# YOLO dataset configuration for Google Colab
path: /content/data/yolo  # dataset root dir
train: train/images       # train images (relative to 'path')
val: val/images          # val images (relative to 'path')
test: test/images        # test images (optional, relative to 'path')

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

    yaml_path = "/content/data/yolo/dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)

    print(f"✓ Created dataset.yaml at {yaml_path}")

def setup_google_drive_integration():
    """Setup Google Drive integration for persistence"""

    print(f"\n💾 Google Drive Integration (Optional but Recommended)")
    print("=" * 55)
    print("For persistent storage across sessions, you can mount Google Drive:")
    print()
    print("# Mount Google Drive")
    print("from google.colab import drive")
    print("drive.mount('/content/drive')")
    print()
    print("# Copy project to Drive (for persistence)")
    print("!cp -r /content/table_detection /content/drive/MyDrive/")
    print()
    print("# Work from Drive location")
    print("%cd /content/drive/MyDrive/table_detection")
    print()
    print("Benefits:")
    print("• Files persist across Colab sessions")
    print("• Easy sharing and backup")
    print("• Access from any device")

def print_colab_usage_info():
    """Print Colab-specific usage information"""

    print(f"\n🎯 COLAB QUICK START COMMANDS:")
    print("=" * 50)
    print("# 1. Install dependencies")
    print("!pip install ultralytics pyyaml pillow tqdm")
    print()
    print("# 2. Create sample dataset")
    print("!python create_sample_dataset.py")
    print()
    print("# 3. Convert to YOLO format")
    print("!python scripts/data_converter.py configs/data_config.yaml")
    print()
    print("# 4. Start training")
    print("!python train_table_detection.py")
    print()

    print(f"\n🗂️ COLAB DIRECTORY STRUCTURE:")
    print("=" * 30)
    print("/content/data/             - Training data")
    print("/content/checkpoints/      - Model checkpoints")
    print("/content/logs/             - Training logs")
    print("/content/results/          - Training results")
    print("/content/drive/MyDrive/    - Google Drive (if mounted)")
    print()

    print(f"\n💡 COLAB TIPS:")
    print("=" * 15)
    print("• 12-hour session limit - save work frequently")
    print("• Free GPU access with T4 (16GB)")
    print("• Mount Google Drive for persistence")
    print("• Use !nvidia-smi to monitor GPU")
    print("• Download models before session expires")
    print("• Use %tensorboard --logdir logs for monitoring")
    print("• Runtime → Factory reset runtime to clear memory")
    print()

    print(f"⚡ PERFORMANCE OPTIMIZATION:")
    print("=" * 30)
    print("• Use batch_size=8-16 for T4 GPU")
    print("• Set workers=2 for data loading")
    print("• Enable mixed precision: amp=True")
    print("• Monitor memory: !nvidia-smi")
    print("• Use smaller image sizes if memory limited")

def setup_colab_monitoring():
    """Setup monitoring tools for Colab"""

    print(f"\n📊 Setting up monitoring tools...")

    # Create monitoring script
    monitor_script = """
# GPU monitoring
import torch
import psutil
import GPUtil

def print_system_info():
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    # CPU info
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f}GB")

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU available")

    print("=" * 50)

if __name__ == "__main__":
    print_system_info()
"""

    with open("/content/system_monitor.py", "w") as f:
        f.write(monitor_script)

    print("✓ Created system monitoring script: /content/system_monitor.py")
    print("Run with: !python /content/system_monitor.py")

def verify_colab_setup():
    """Verify Colab setup is working correctly"""

    print(f"\n🔍 Verifying Colab setup...")

    # Check directories
    required_dirs = [
        "/content/data",
        "/content/checkpoints",
        "/content/logs",
        "/content/results"
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
            print("⚠️ No GPU detected - enable in Runtime → Change runtime type → GPU")
    except ImportError:
        print("⚠️ PyTorch not installed")

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/content")
    free_gb = free // (1024**3)
    print(f"💾 Available storage: {free_gb}GB")

    if missing_dirs:
        print(f"\n❌ Setup incomplete - missing directories: {missing_dirs}")
        return False
    else:
        print(f"\n✅ Colab setup verified successfully!")
        return True

def main():
    """Main setup function"""

    try:
        # Run setup
        success = colab_quickstart()

        if success:
            # Setup monitoring
            setup_colab_monitoring()

            # Verify setup
            if verify_colab_setup():
                print(f"\n🎉 Google Colab environment ready for table detection training!")
                print(f"\n📝 Remember to:")
                print(f"• Enable GPU if not already done")
                print(f"• Mount Google Drive for persistence")
                print(f"• Save work frequently (12-hour session limit)")
                return True
            else:
                print(f"\n⚠️ Setup completed but verification failed")
                return False
        else:
            print(f"\n❌ Colab setup failed!")
            return False

    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)