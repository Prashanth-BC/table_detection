#!/usr/bin/env python3
import os
import json
import yaml
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

class TableDetectionTrainer:
    def __init__(self, config_path="configs/model_config.yaml"):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()
        self.detect_hardware()
        self.model = None

    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.train_config = self.config['train']

    def setup_directories(self):
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
        device_info = {'device': 'cpu', 'gpu_available': False, 'gpu_count': 0}
        if torch.cuda.is_available():
            device_info.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'device': 'cuda'
            })
        if self.train_config['device'] == 'auto':
            self.train_config['device'] = device_info['device']
        self.device_info = device_info
        self.update_progress({'hardware': device_info})

    def update_progress(self, updates):
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

    def find_latest_checkpoint(self):
        from scripts.platform_detector import get_platform_config
        platform_config = get_platform_config()
        checkpoint_dir = Path(platform_config.config['checkpoints_dir'])

        for checkpoint in ["last.pt", "best.pt"]:
            path = checkpoint_dir / checkpoint
            if path.exists():
                return str(path)

        pt_files = list(checkpoint_dir.glob("*.pt"))
        if pt_files:
            return str(max(pt_files, key=os.path.getctime))
        return None

    def setup_model(self, resume_from=None):
        if resume_from and os.path.exists(resume_from):
            self.model = YOLO(resume_from)
        else:
            self.model = YOLO("yolo11n.pt")

    def train(self, dataset_path=None, resume=True):
        if dataset_path is None:
            from scripts.platform_detector import get_platform_config
            platform_config = get_platform_config()
            dataset_path = platform_config.get_data_paths()['dataset_yaml']

        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return False

        resume_from = self.find_latest_checkpoint() if resume else None
        self.setup_model(resume_from)

        self.update_progress({
            'current_stage': 'training',
            'training': {
                'started_at': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'resume_from': resume_from
            }
        })

        try:
            from scripts.platform_detector import get_platform_config
            platform_config = get_platform_config()

            train_params = {
                'data': dataset_path,
                'epochs': 30,
                'batch': 8,
                'imgsz': 640,
                'device': self.train_config['device'],
                'workers': 2,
                'patience': 10,
                'save_period': 5,
                'project': platform_config.config['results_dir'],
                'name': 'table_detection',
                'exist_ok': True,
                'resume': bool(resume_from),
                'amp': True
            }

            results = self.model.train(**train_params)

            results_path = f"{platform_config.config['results_dir']}/table_detection/weights"
            if os.path.exists(f"{results_path}/best.pt"):
                import shutil
                shutil.copy2(f"{results_path}/best.pt", f"{platform_config.config['checkpoints_dir']}/best.pt")
                shutil.copy2(f"{results_path}/last.pt", f"{platform_config.config['checkpoints_dir']}/last.pt")

            self.update_progress({
                'current_stage': 'completed',
                'training': {
                    'completed_at': datetime.now().isoformat(),
                    'final_model_path': f"{platform_config.config['checkpoints_dir']}/best.pt"
                }
            })
            return True

        except Exception as e:
            self.update_progress({
                'stages': {'training': {'status': 'failed', 'description': str(e)}}
            })
            print(f"Training failed: {e}")
            return False

    def evaluate(self, model_path=None, dataset_path=None):
        from scripts.platform_detector import get_platform_config
        platform_config = get_platform_config()

        if model_path is None:
            model_path = f"{platform_config.config['checkpoints_dir']}/best.pt"
        if dataset_path is None:
            dataset_path = platform_config.get_data_paths()['dataset_yaml']

        if not os.path.exists(model_path):
            return None

        model = YOLO(model_path)
        results = model.val(
            data=dataset_path,
            batch=8,
            imgsz=640,
            device=self.train_config['device'],
            project=platform_config.config['results_dir'],
            name='evaluation',
            exist_ok=True
        )

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
        return results

def main():
    from scripts.platform_detector import get_platform_config
    platform_config = get_platform_config()

    trainer = TableDetectionTrainer()
    dataset_path = platform_config.get_data_paths()['dataset_yaml']

    if not os.path.exists(dataset_path):
        print("Dataset not found. Run: python scripts/data_converter.py")
        return

    success = trainer.train(resume=True)
    if success:
        trainer.evaluate()
        print(f"Training complete. Model: {platform_config.config['checkpoints_dir']}/best.pt")
    else:
        print("Training failed")

if __name__ == "__main__":
    main()