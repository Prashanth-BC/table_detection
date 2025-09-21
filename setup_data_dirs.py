#!/usr/bin/env python3
"""
Setup data directories in /content for faster access during training.
"""

import os
from pathlib import Path

def setup_data_directories():
    """Create necessary data directories in /content"""

    # Data directories in /content (faster access)
    data_dirs = [
        "/content/data/raw",
        "/content/data/processed",
        "/content/data/yolo/train/images",
        "/content/data/yolo/train/labels",
        "/content/data/yolo/val/images",
        "/content/data/yolo/val/labels",
        "/content/data/yolo/test/images",
        "/content/data/yolo/test/labels",
        "/content/models",
        "/content/checkpoints",
        "/content/logs",
        "/content/results"
    ]

    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")

    print(f"\n✅ All data directories created in /content for faster access!")
    print("📁 Data structure:")
    print("   /content/data/raw/     - Raw downloaded datasets")
    print("   /content/data/yolo/    - YOLO format training data")
    print("   /content/checkpoints/  - Model checkpoints")
    print("   /content/logs/         - Training logs")
    print("   /content/results/      - Training results")

if __name__ == "__main__":
    setup_data_directories()