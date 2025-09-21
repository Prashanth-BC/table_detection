#!/usr/bin/env python3
"""
Platform detection utility for Google Colab vs Kaggle environments.
Automatically configures paths and settings based on detected platform.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

class PlatformConfig:
    """Platform-specific configuration manager"""

    def __init__(self):
        self.platform = self.detect_platform()
        self.config = self.get_platform_config()

    def detect_platform(self) -> str:
        """Detect if running on Google Colab or Kaggle"""

        # Check for Kaggle environment
        if os.path.exists('/kaggle'):
            return 'kaggle'

        # Check for Google Colab environment
        elif os.path.exists('/content'):
            return 'colab'

        # Default to local development
        else:
            return 'local'

    def get_platform_config(self) -> Dict:
        """Get platform-specific configuration"""

        configs = {
            'kaggle': {
                'data_root': '/kaggle/tmp/data',
                'working_dir': '/kaggle/working',
                'input_dir': '/kaggle/input',
                'project_dir': '/kaggle/working/table_detection',
                'checkpoints_dir': '/kaggle/working/checkpoints',
                'logs_dir': '/kaggle/working/logs',
                'results_dir': '/kaggle/working/results',
                'max_storage_gb': 73,
                'gpu_quota_hours': 30,
                'session_limit_hours': 9,
                'advantages': ['Higher GPU quota', 'Better for production training']
            },

            'colab': {
                'data_root': '/content/data',
                'working_dir': '/content',
                'drive_dir': '/content/drive/MyDrive',
                'project_dir': '/content/drive/MyDrive/table_detection',
                'checkpoints_dir': '/content/checkpoints',
                'logs_dir': '/content/logs',
                'results_dir': '/content/results',
                'max_storage_gb': 79,
                'gpu_quota_hours': 12,
                'session_limit_hours': 12,
                'advantages': ['Easy Google Drive integration', 'Good for development']
            },

            'local': {
                'data_root': './data',
                'working_dir': '.',
                'project_dir': './table_detection',
                'checkpoints_dir': './checkpoints',
                'logs_dir': './logs',
                'results_dir': './results',
                'max_storage_gb': 'unlimited',
                'gpu_quota_hours': 'unlimited',
                'session_limit_hours': 'unlimited',
                'advantages': ['Full control', 'No time limits']
            }
        }

        return configs[self.platform]

    def get_data_paths(self) -> Dict[str, str]:
        """Get platform-specific data paths"""

        base_data = self.config['data_root']

        return {
            'raw_data': f"{base_data}/raw",
            'processed_data': f"{base_data}/processed",
            'yolo_data': f"{base_data}/yolo",
            'train_images': f"{base_data}/yolo/train/images",
            'train_labels': f"{base_data}/yolo/train/labels",
            'val_images': f"{base_data}/yolo/val/images",
            'val_labels': f"{base_data}/yolo/val/labels",
            'test_images': f"{base_data}/yolo/test/images",
            'test_labels': f"{base_data}/yolo/test/labels",
            'dataset_yaml': f"{base_data}/yolo/dataset.yaml"
        }

    def setup_directories(self):
        """Create all necessary directories for the detected platform"""

        print(f"🔍 Detected platform: {self.platform.upper()}")
        print(f"📁 Setting up directories for {self.platform}...")

        # Get all paths that need to be created
        paths_to_create = []

        # Add data paths
        data_paths = self.get_data_paths()
        paths_to_create.extend(data_paths.values())

        # Add working directories
        paths_to_create.extend([
            self.config['checkpoints_dir'],
            self.config['logs_dir'],
            self.config['results_dir']
        ])

        # Create directories
        created_count = 0
        for path in paths_to_create:
            Path(path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {path}")
            created_count += 1

        # Platform-specific setup
        if self.platform == 'kaggle':
            # Create symlink for easier access
            working_data = Path("/kaggle/working/data")
            if not working_data.exists():
                working_data.symlink_to("/kaggle/tmp/data")
                print(f"✓ Created symlink: {working_data} -> /kaggle/tmp/data")

        print(f"\n✅ {self.platform.upper()} environment setup complete!")
        print(f"📊 Created {created_count} directories")
        print(f"💾 Max storage: {self.config['max_storage_gb']}GB")
        print(f"⏰ Session limit: {self.config['session_limit_hours']} hours")

        return True

    def get_training_config_updates(self) -> Dict:
        """Get platform-specific training configuration updates"""

        updates = {
            'kaggle': {
                'data_path': '/kaggle/tmp/data/yolo/dataset.yaml',
                'project_path': '/kaggle/working/results',
                'device': 'auto',  # Kaggle auto-detects GPU
                'workers': 2,  # Conservative for Kaggle
                'cache': False,  # Disable caching to save memory
            },

            'colab': {
                'data_path': '/content/data/yolo/dataset.yaml',
                'project_path': '/content/results',
                'device': 'auto',  # Colab auto-detects GPU
                'workers': 2,  # Conservative for Colab
                'cache': False,  # Disable caching to save memory
            },

            'local': {
                'data_path': './data/yolo/dataset.yaml',
                'project_path': './results',
                'device': 'auto',
                'workers': 4,  # Can use more workers locally
                'cache': True,  # Enable caching if sufficient memory
            }
        }

        return updates[self.platform]

    def print_platform_info(self):
        """Print detailed platform information"""

        print("=" * 60)
        print(f"🚀 PLATFORM: {self.platform.upper()}")
        print("=" * 60)
        print(f"📁 Data root: {self.config['data_root']}")
        print(f"💾 Max storage: {self.config['max_storage_gb']}GB")
        print(f"⏰ Session limit: {self.config['session_limit_hours']} hours")
        print(f"🎯 GPU quota: {self.config['gpu_quota_hours']} hours")

        print(f"\n✨ Advantages:")
        for advantage in self.config['advantages']:
            print(f"   • {advantage}")

        print(f"\n📂 Key directories:")
        print(f"   • Project: {self.config['project_dir']}")
        print(f"   • Data: {self.config['data_root']}")
        print(f"   • Checkpoints: {self.config['checkpoints_dir']}")
        print(f"   • Results: {self.config['results_dir']}")
        print("=" * 60)

def get_platform_config() -> PlatformConfig:
    """Get platform configuration instance"""
    return PlatformConfig()

def main():
    """Main function for testing platform detection"""
    config = get_platform_config()
    config.print_platform_info()

    # Setup directories
    config.setup_directories()

    print(f"\n🔧 Training config updates:")
    updates = config.get_training_config_updates()
    for key, value in updates.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    main()