#!/usr/bin/env python3
"""
Train YOLO11n model for table structure detection.
Supports sample dataset training with GPU acceleration and checkpointing.
"""

import os
import sys
import json
import yaml
import torch
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TableDetectionTrainer:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize the trainer with configuration"""
        self.config_path = config_path
        self.load_config()
        self.setup_directories()
        self.detect_hardware()
        self.model = None

    def load_config(self):
        """Load training configuration"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.train_config = self.config['train']
        self.val_config = self.config['val']
        self.checkpoint_config = self.config['checkpoint']

    def setup_directories(self):
        """Setup required directories"""
        # Use platform-specific directories
        from scripts.platform_detector import get_platform_config
        platform_config = get_platform_config()
        directories = [
            platform_config.config['logs_dir'],
            platform_config.config['checkpoints_dir'],
            platform_config.config['results_dir']
        ]
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)

    def detect_hardware(self):
        """Detect available hardware for training"""
        device_info = {
            'device': 'cpu',
            'gpu_available': False,
            'tpu_available': False,
            'gpu_count': 0,
            'gpu_memory': 0
        }

        # Check for CUDA GPUs
        if torch.cuda.is_available():
            device_info['gpu_available'] = True
            device_info['gpu_count'] = torch.cuda.device_count()
            device_info['device'] = 'cuda'

            if device_info['gpu_count'] > 0:
                device_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"Found {device_info['gpu_count']} GPU(s), Memory: {device_info['gpu_memory']:.1f}GB")

        # Check for TPU (Google Colab/Kaggle)
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device_info['tpu_available'] = True
            device_info['device'] = 'tpu'
            logger.info("TPU detected and available")
        except ImportError:
            pass

        # Update device in config
        if self.train_config['device'] == 'auto':
            self.train_config['device'] = device_info['device']

        self.device_info = device_info
        logger.info(f"Using device: {device_info['device']}")

        # Update progress file
        self.update_progress({'hardware': device_info})

    def update_progress(self, updates: dict):
        """Update progress tracking file"""
        progress_file = "progress.json"

        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {}

        progress.update(updates)
        progress['last_updated'] = datetime.now().isoformat()

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def find_latest_checkpoint(self) -> str:
        """Find the latest checkpoint for resuming training"""
        # Use platform-specific checkpoint directory
        from scripts.platform_detector import get_platform_config
        platform_config = get_platform_config()
        checkpoint_dir = Path(platform_config.config['checkpoints_dir'])

        # Look for last.pt (latest checkpoint)
        last_checkpoint = checkpoint_dir / "last.pt"
        if last_checkpoint.exists():
            logger.info(f"Found latest checkpoint: {last_checkpoint}")
            return str(last_checkpoint)

        # Look for best.pt (best checkpoint)
        best_checkpoint = checkpoint_dir / "best.pt"
        if best_checkpoint.exists():
            logger.info(f"Found best checkpoint: {best_checkpoint}")
            return str(best_checkpoint)

        # Look for any .pt files
        pt_files = list(checkpoint_dir.glob("*.pt"))
        if pt_files:
            latest_file = max(pt_files, key=os.path.getctime)
            logger.info(f"Found checkpoint: {latest_file}")
            return str(latest_file)

        return None

    def setup_model(self, resume_from: str = None):
        """Setup YOLO model for training"""
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.model = YOLO(resume_from)
        else:
            logger.info("Starting training from scratch with YOLO11n")
            self.model = YOLO("yolo11n.pt")  # Load pretrained weights

    def train(self, dataset_path: str = None, resume: bool = True):
        """Train the table detection model"""
        logger.info("Starting table structure detection training...")

        # Use platform-specific dataset path if not provided
        if dataset_path is None:
            from scripts.platform_detector import get_platform_config
            platform_config = get_platform_config()
            dataset_path = platform_config.get_data_paths()['dataset_yaml']

        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at {dataset_path}")
            return False

        # Setup model
        resume_from = None
        if resume:
            resume_from = self.find_latest_checkpoint()

        self.setup_model(resume_from)

        # Update progress
        self.update_progress({
            'current_stage': 'training',
            'stages': {
                'training': {
                    'status': 'in_progress',
                    'progress': 0,
                    'description': 'Training YOLO11n model'
                }
            },
            'training': {
                'started_at': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'resume_from': resume_from
            }
        })

        try:
            # Reduced training parameters for sample dataset
            train_params = {
                'data': dataset_path,
                'epochs': 30,  # Reduced for sample dataset
                'batch': 8,    # Reduced batch size for sample data
                'imgsz': 640,
                'device': self.train_config['device'],
                'workers': 2,  # Reduced workers
                'patience': 10,
                'save_period': 5,
                'project': platform_config.config['results_dir'],
                'name': 'table_detection',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'save': True,
                'cache': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': bool(resume_from),
                'amp': True,  # Automatic Mixed Precision
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': False,
                # Data augmentation (reduced for sample data)
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
            }

            logger.info(f"Training parameters:")
            logger.info(f"  Epochs: {train_params['epochs']}")
            logger.info(f"  Batch size: {train_params['batch']}")
            logger.info(f"  Device: {train_params['device']}")
            logger.info(f"  Image size: {train_params['imgsz']}")

            # Start training
            logger.info("Starting YOLO training...")
            results = self.model.train(**train_params)

            # Copy model to checkpoints directory for easier access
            results_path = f"{platform_config.config['results_dir']}/table_detection/weights"
            if os.path.exists(f"{results_path}/best.pt"):
                import shutil
                shutil.copy2(
                    f"{results_path}/best.pt",
                    f"{platform_config.config['checkpoints_dir']}/best.pt"
                )
                shutil.copy2(
                    f"{results_path}/last.pt",
                    f"{platform_config.config['checkpoints_dir']}/last.pt"
                )
                logger.info(f"Models copied to {platform_config.config['checkpoints_dir']}/")

            # Update progress
            self.update_progress({
                'current_stage': 'completed',
                'stages': {
                    'training': {
                        'status': 'completed',
                        'progress': 100,
                        'description': 'Training completed successfully'
                    }
                },
                'training': {
                    'completed_at': datetime.now().isoformat(),
                    'final_model_path': f"{platform_config.config['checkpoints_dir']}/best.pt",
                    'results_path': f"{platform_config.config['results_dir']}/table_detection"
                }
            })

            logger.info(f"Training completed successfully!")
            logger.info(f"Best model saved to: {platform_config.config['checkpoints_dir']}/best.pt")
            logger.info(f"Results saved to: {platform_config.config['results_dir']}/table_detection")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.update_progress({
                'stages': {
                    'training': {
                        'status': 'failed',
                        'progress': 0,
                        'description': f'Training failed: {str(e)}'
                    }
                }
            })
            return False

    def evaluate(self, model_path: str = None, dataset_path: str = None):
        """Evaluate the trained model"""
        # Use platform-specific paths if not provided
        from scripts.platform_detector import get_platform_config
        platform_config = get_platform_config()

        if model_path is None:
            model_path = f"{platform_config.config['checkpoints_dir']}/best.pt"

        if dataset_path is None:
            dataset_path = platform_config.get_data_paths()['dataset_yaml']

        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None

        logger.info(f"Evaluating model: {model_path}")

        # Load model
        model = YOLO(model_path)

        # Run validation
        results = model.val(
            data=dataset_path,
            batch=8,
            imgsz=640,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=False,
            device=self.train_config['device'],
            save_json=True,
            project=platform_config.config['results_dir'],
            name='evaluation',
            exist_ok=True
        )

        # Update progress
        eval_results = {
            'map50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            'map': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
        }

        self.update_progress({
            'evaluation': {
                'completed_at': datetime.now().isoformat(),
                'model_path': model_path,
                'results': eval_results
            }
        })

        logger.info(f"Evaluation completed:")
        logger.info(f"  mAP50: {eval_results['map50']:.3f}")
        logger.info(f"  mAP: {eval_results['map']:.3f}")
        logger.info(f"  Precision: {eval_results['precision']:.3f}")
        logger.info(f"  Recall: {eval_results['recall']:.3f}")

        return results

def main():
    """Main execution function"""
    print("🚀 YOLO11n Table Structure Detection Training")
    print("=" * 50)

    # Setup platform-specific paths
    from scripts.platform_detector import get_platform_config
    platform_config = get_platform_config()
    platform_config.print_platform_info()

    # Create trainer
    trainer = TableDetectionTrainer()

    # Check dataset
    dataset_path = platform_config.get_data_paths()['dataset_yaml']
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Run: python scripts/data_converter.py first")
        return

    # Start training
    print("Starting training with sample dataset...")
    success = trainer.train(resume=True)

    if success:
        print("✅ Training completed successfully!")

        # Run evaluation
        print("Starting evaluation...")
        results = trainer.evaluate()

        if results:
            print("✅ Evaluation completed successfully!")
        else:
            print("⚠️ Evaluation failed or skipped")

        print("\n🎯 Next steps:")
        print(f"1. Check results in {platform_config.config['results_dir']}/table_detection/")
        print(f"2. Use model for inference: YOLO('{platform_config.config['checkpoints_dir']}/best.pt')")
        print("3. Download real datasets for better accuracy")

    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()