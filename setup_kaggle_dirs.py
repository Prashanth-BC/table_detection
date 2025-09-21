#!/usr/bin/env python3
"""
Setup data directories for Kaggle environment.
Optimized for Kaggle's /kaggle/tmp/ storage for large datasets.
"""

import os
from pathlib import Path

def setup_kaggle_directories():
    """Create necessary data directories for Kaggle environment"""

    # Kaggle-specific data directories
    data_dirs = [
        # Large temporary storage in /kaggle/tmp (73GB available)
        "/kaggle/tmp/data/raw",
        "/kaggle/tmp/data/processed",
        "/kaggle/tmp/data/yolo/train/images",
        "/kaggle/tmp/data/yolo/train/labels",
        "/kaggle/tmp/data/yolo/val/images",
        "/kaggle/tmp/data/yolo/val/labels",
        "/kaggle/tmp/data/yolo/test/images",
        "/kaggle/tmp/data/yolo/test/labels",

        # Working directory storage (session-persistent)
        "/kaggle/working/checkpoints",
        "/kaggle/working/logs",
        "/kaggle/working/results",
        "/kaggle/working/models",
        "/kaggle/working/final_models"
    ]

    print("🚀 Setting up Kaggle directories for table detection training...")

    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")

    # Create symlinks for easier access
    working_data = Path("/kaggle/working/data")
    if not working_data.exists():
        working_data.symlink_to("/kaggle/tmp/data")
        print(f"✓ Created symlink: {working_data} -> /kaggle/tmp/data")

    print(f"\n✅ Kaggle environment setup complete!")
    print("📁 Kaggle storage structure:")
    print("   /kaggle/tmp/data/      - Large datasets (73GB available)")
    print("   /kaggle/working/       - Project files and checkpoints")
    print("   /kaggle/input/         - Input datasets")
    print("\n💡 Storage tips for Kaggle:")
    print("   - Use /kaggle/tmp/ for large temporary files")
    print("   - Save important results in /kaggle/working/")
    print("   - Upload trained models as Kaggle datasets for reuse")

if __name__ == "__main__":
    setup_kaggle_directories()