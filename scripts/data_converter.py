#!/usr/bin/env python3
"""
Convert various table detection datasets to YOLO format.
Handles COCO format annotations and converts them to YOLO txt format.
Data stored in /content for faster access.
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataConverter:
    def __init__(self, config_path: str, raw_dir: str = None, yolo_dir: str = None):
        # Auto-detect platform and set paths
        try:
            from .platform_detector import get_platform_config
            platform_config = get_platform_config()
            data_paths = platform_config.get_data_paths()

            self.raw_dir = Path(raw_dir) if raw_dir else Path(data_paths['raw_data'])
            self.yolo_dir = Path(yolo_dir) if yolo_dir else Path(data_paths['yolo_data'])
        except ImportError:
            # Fallback to default paths
            self.raw_dir = Path(raw_dir) if raw_dir else Path("/content/data/raw")
            self.yolo_dir = Path(yolo_dir) if yolo_dir else Path("/content/data/yolo")

        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.class_mapping = self.config['yolo_classes']
        self.split_ratios = self.config['split_ratios']

    def update_progress(self, details: Dict):
        """Update progress tracking file"""
        progress_file = "progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {"datasets": {}}

        progress["datasets"].update(details)

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def coco_to_yolo_bbox(self, coco_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert COCO bbox format [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized)"""
        x, y, w, h = coco_bbox

        # Convert to center coordinates and normalize
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        return [x_center, y_center, width, height]

    def create_yolo_annotation(self, annotations: List[Dict], img_width: int, img_height: int,
                             category_mapping: Dict[int, int]) -> List[str]:
        """Create YOLO format annotation lines"""
        yolo_lines = []

        for ann in annotations:
            category_id = ann['category_id']

            # Map to YOLO class ID
            if category_id in category_mapping:
                yolo_class_id = category_mapping[category_id]

                # Convert bbox
                coco_bbox = ann['bbox']
                yolo_bbox = self.coco_to_yolo_bbox(coco_bbox, img_width, img_height)

                # Create YOLO line: class_id x_center y_center width height
                line = f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}"
                yolo_lines.append(line)

        return yolo_lines

    def split_dataset(self, image_files: List[str]) -> Dict[str, List[str]]:
        """Split dataset into train/val/test sets"""
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(image_files)

        total = len(image_files)
        train_end = int(total * self.split_ratios['train'])
        val_end = train_end + int(total * self.split_ratios['val'])

        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }

        return splits

    def convert_sample_dataset(self) -> bool:
        """Convert sample dataset to YOLO format"""
        dataset_name = "sample_tables"
        dataset_dir = self.raw_dir / dataset_name

        if not dataset_dir.exists():
            logger.warning(f"Sample dataset not found at {dataset_dir}")
            return False

        logger.info(f"Converting {dataset_name} to YOLO format...")

        # Load annotations
        ann_file = dataset_dir / "annotations.json"
        if not ann_file.exists():
            logger.warning(f"Annotations file not found: {ann_file}")
            return False

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # Create category mapping (COCO category_id -> YOLO class_id)
        category_mapping = {}
        for cat in coco_data['categories']:
            coco_id = cat['id']
            yolo_id = coco_id  # Direct mapping for sample data
            category_mapping[coco_id] = yolo_id

        # Process images
        processed_count = 0
        for img_info in tqdm(coco_data['images'], desc="Converting images"):
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            # Get annotations for this image
            img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

            if not img_annotations:
                continue

            # Create YOLO annotation
            yolo_lines = self.create_yolo_annotation(img_annotations, img_width, img_height, category_mapping)

            if yolo_lines:
                # For sample dataset, split into train/val
                if processed_count < 40:  # First 40 for training
                    split = 'train'
                else:  # Last 10 for validation
                    split = 'val'

                # Copy image (create a dummy image if not exists)
                src_img_path = dataset_dir / img_filename
                dst_img_path = self.yolo_dir / split / 'images' / img_filename

                if not src_img_path.exists():
                    # Create a dummy image for demonstration
                    dummy_img = Image.new('RGB', (img_width, img_height), color='white')
                    dummy_img.save(dst_img_path)
                else:
                    shutil.copy2(src_img_path, dst_img_path)

                # Save YOLO annotation
                label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = self.yolo_dir / split / 'labels' / label_filename

                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                processed_count += 1

        logger.info(f"Converted {processed_count} images from {dataset_name}")
        return processed_count > 0

    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        dataset_yaml = {
            'path': str(self.yolo_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.values())
        }

        yaml_path = self.yolo_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        logger.info(f"Created dataset.yaml at {yaml_path}")

    def get_dataset_stats(self) -> Dict:
        """Get statistics about the converted dataset"""
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'splits': {}
        }

        for split in ['train', 'val', 'test']:
            split_dir = self.yolo_dir / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'

            if images_dir.exists():
                image_count = len(list(images_dir.glob('*')))
                label_count = len(list(labels_dir.glob('*.txt')))

                # Count total annotations
                total_anns = 0
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        total_anns += len(f.readlines())

                stats['splits'][split] = {
                    'images': image_count,
                    'labels': label_count,
                    'annotations': total_anns
                }

                stats['total_images'] += image_count
                stats['total_annotations'] += total_anns

        return stats

    def convert_all_datasets(self) -> bool:
        """Convert all available datasets to YOLO format"""
        success = False

        # Convert sample dataset first
        if self.convert_sample_dataset():
            success = True

        # Create dataset.yaml
        self.create_dataset_yaml()

        # Get statistics
        stats = self.get_dataset_stats()
        logger.info(f"Dataset conversion completed. Stats: {stats}")

        # Update progress
        self.update_progress({
            'total_images': stats['total_images'],
            'total_annotations': stats['total_annotations'],
            'conversion_stats': stats
        })

        return success

def main():
    """Main execution function"""
    config_path = "configs/dataset_config.yaml"
    converter = DataConverter(config_path)

    logger.info("Starting dataset conversion to YOLO format...")
    success = converter.convert_all_datasets()

    if success:
        logger.info("Dataset conversion completed successfully!")
    else:
        logger.error("Dataset conversion failed!")

if __name__ == "__main__":
    main()