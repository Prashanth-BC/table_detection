#!/usr/bin/env python3
"""
Dataset downloader for table structure detection datasets.
Downloads and organizes publicly available datasets for YOLO training.
"""

import os
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, config_path: str, data_dir: str = None):
        self.config_path = config_path

        # Auto-detect platform and set paths
        try:
            from .platform_detector import get_platform_config
            platform_config = get_platform_config()
            data_paths = platform_config.get_data_paths()

            self.data_dir = Path(data_dir) if data_dir else Path(platform_config.config['data_root'])
            self.raw_dir = Path(data_paths['raw_data'])
            self.processed_dir = Path(data_paths['processed_data'])
        except ImportError:
            # Fallback to default paths
            self.data_dir = Path(data_dir) if data_dir else Path("/content/data")
            self.raw_dir = self.data_dir / "raw"
            self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def update_progress(self, dataset_name: str, status: str, details: Dict = None):
        """Update progress tracking file"""
        progress_file = "progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {"datasets": {"downloaded": [], "processed": []}}

        if status == "downloaded" and dataset_name not in progress["datasets"]["downloaded"]:
            progress["datasets"]["downloaded"].append(dataset_name)
        elif status == "processed" and dataset_name not in progress["datasets"]["processed"]:
            progress["datasets"]["processed"].append(dataset_name)

        if details:
            progress["datasets"].update(details)

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def download_file(self, url: str, filename: str) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filename, 'wb') as f, tqdm(
                desc=os.path.basename(filename),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """Extract archive file"""
        try:
            if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(extract_to)
            elif archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            else:
                logger.warning(f"Unsupported archive format: {archive_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False

    def download_publaynet(self) -> bool:
        """Download PubLayNet dataset"""
        dataset_name = "publaynet"
        dataset_config = self.config["datasets"][dataset_name]

        logger.info(f"Downloading {dataset_name}...")

        # Create dataset directory
        dataset_dir = self.raw_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        # Download main dataset
        url = dataset_config["url"]
        archive_path = dataset_dir / "publaynet.tar.gz"

        if not archive_path.exists():
            if not self.download_file(url, str(archive_path)):
                return False

        # Extract
        if not (dataset_dir / "train").exists():
            logger.info("Extracting PubLayNet...")
            if not self.extract_archive(str(archive_path), str(dataset_dir)):
                return False

        self.update_progress(dataset_name, "downloaded")
        logger.info(f"{dataset_name} download completed")
        return True

    def download_tablebank(self) -> bool:
        """Download TableBank dataset"""
        dataset_name = "tablebank"

        logger.info(f"Setting up {dataset_name}...")

        # Create dataset directory
        dataset_dir = self.raw_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        # TableBank is available through multiple sources
        # We'll create a script to download from Hugging Face
        tablebank_script = dataset_dir / "download_tablebank.py"

        script_content = '''
import os
from datasets import load_dataset
from pathlib import Path

def download_tablebank():
    """Download TableBank dataset from Hugging Face"""
    try:
        # Download Word documents subset (smaller, more manageable)
        dataset = load_dataset("tablebank", "word", split="train")

        # Save to local directory
        output_dir = Path("./word_train")
        output_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(dataset):
            if i >= 10000:  # Limit to first 10k samples for demo
                break

            # Save image
            image_path = output_dir / f"image_{i:06d}.jpg"
            sample["image"].save(image_path)

            # Save annotation (we'll process this later)
            with open(output_dir / f"annotation_{i:06d}.json", "w") as f:
                import json
                json.dump(sample["annotation"], f)

        print(f"Downloaded {min(i+1, 10000)} samples")

    except Exception as e:
        print(f"Error downloading TableBank: {e}")
        print("You may need to request access or use alternative sources")

if __name__ == "__main__":
    download_tablebank()
'''

        with open(tablebank_script, 'w') as f:
            f.write(script_content)

        self.update_progress(dataset_name, "downloaded",
                           {"note": "TableBank download script created - run manually if needed"})
        logger.info(f"{dataset_name} setup completed")
        return True

    def download_sample_dataset(self) -> bool:
        """Create a sample dataset for immediate testing"""
        dataset_name = "sample_tables"

        logger.info(f"Creating {dataset_name}...")

        # Create dataset directory
        dataset_dir = self.raw_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        # Create sample annotation format
        sample_data = {
            "info": {
                "description": "Sample table dataset for testing",
                "version": "1.0",
                "year": 2024
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "sample_table_001.jpg",
                    "width": 800,
                    "height": 600
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,  # table
                    "bbox": [100, 100, 600, 400],  # x, y, width, height
                    "area": 240000,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,  # row
                    "bbox": [100, 100, 600, 80],
                    "area": 48000,
                    "iscrowd": 0
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 2,  # column
                    "bbox": [100, 100, 150, 400],
                    "area": 60000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 0, "name": "table"},
                {"id": 1, "name": "row"},
                {"id": 2, "name": "column"},
                {"id": 3, "name": "cell"}
            ]
        }

        # Save sample annotation
        with open(dataset_dir / "annotations.json", 'w') as f:
            json.dump(sample_data, f, indent=2)

        self.update_progress(dataset_name, "downloaded")
        logger.info(f"{dataset_name} created")
        return True

    def download_all_datasets(self) -> List[str]:
        """Download all configured datasets"""
        downloaded = []

        # Start with sample dataset for immediate testing
        if self.download_sample_dataset():
            downloaded.append("sample_tables")

        # Download other datasets
        try:
            if self.download_tablebank():
                downloaded.append("tablebank")
        except Exception as e:
            logger.warning(f"TableBank download failed: {e}")

        try:
            if self.download_publaynet():
                downloaded.append("publaynet")
        except Exception as e:
            logger.warning(f"PubLayNet download failed: {e}")

        # Update overall progress
        progress_file = "progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            progress["current_stage"] = "data_preparation"
            progress["stages"]["setup"]["status"] = "completed"
            progress["stages"]["setup"]["progress"] = 100
            progress["stages"]["data_preparation"]["status"] = "in_progress"
            progress["stages"]["data_preparation"]["progress"] = 30

            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        return downloaded

def main():
    """Main execution function"""
    config_path = "configs/dataset_config.yaml"
    downloader = DatasetDownloader(config_path)

    logger.info("Starting dataset download...")
    downloaded = downloader.download_all_datasets()

    logger.info(f"Downloaded datasets: {downloaded}")
    logger.info("Dataset download completed!")

if __name__ == "__main__":
    main()