#!/usr/bin/env python3
"""
Create sample dataset for testing table structure detection.
Generates synthetic table images with annotations.
"""

import os
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple, Dict

def create_sample_table_image(width: int = 800, height: int = 600,
                            rows: int = 4, cols: int = 3) -> Tuple[Image.Image, List[Dict]]:
    """Create a sample table image with annotations"""

    # Create blank image
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # Calculate cell dimensions
    table_margin = 50
    table_width = width - 2 * table_margin
    table_height = height - 2 * table_margin

    cell_width = table_width // cols
    cell_height = table_height // rows

    # Adjust table size to fit cells exactly
    table_width = cell_width * cols
    table_height = cell_height * rows

    table_x = table_margin
    table_y = table_margin

    annotations = []

    # Draw table border (table annotation)
    table_bbox = [table_x, table_y, table_width, table_height]
    draw.rectangle([table_x, table_y, table_x + table_width, table_y + table_height],
                  outline='black', width=3)

    annotations.append({
        'category_id': 0,  # table
        'bbox': table_bbox,
        'area': table_width * table_height
    })

    # Draw grid and create annotations
    for row in range(rows):
        for col in range(cols):
            x = table_x + col * cell_width
            y = table_y + row * cell_height

            # Draw cell
            draw.rectangle([x, y, x + cell_width, y + cell_height],
                         outline='black', width=1)

            # Add some text content
            try:
                font = ImageFont.load_default()
            except:
                font = None

            text = f"R{row+1}C{col+1}"
            if font:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x + (cell_width - text_width) // 2
                text_y = y + (cell_height - text_height) // 2
                draw.text((text_x, text_y), text, fill='black', font=font)

            # Cell annotation
            annotations.append({
                'category_id': 3,  # cell
                'bbox': [x, y, cell_width, cell_height],
                'area': cell_width * cell_height
            })

    # Add row annotations
    for row in range(rows):
        y = table_y + row * cell_height
        row_bbox = [table_x, y, table_width, cell_height]
        annotations.append({
            'category_id': 1,  # row
            'bbox': row_bbox,
            'area': table_width * cell_height
        })

    # Add column annotations
    for col in range(cols):
        x = table_x + col * cell_width
        col_bbox = [x, table_y, cell_width, table_height]
        annotations.append({
            'category_id': 2,  # column
            'bbox': col_bbox,
            'area': cell_width * table_height
        })

    return image, annotations

def create_sample_dataset(num_images: int = 20, output_dir: str = "/content/data/raw/sample_tables"):
    """Create a sample dataset with multiple table images"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset metadata
    dataset_info = {
        "info": {
            "description": "Sample table dataset for testing YOLO11n table structure detection",
            "version": "1.0",
            "year": 2024,
            "contributor": "Claude Code Assistant",
            "date_created": "2025-09-21"
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "table", "supercategory": "structure"},
            {"id": 1, "name": "row", "supercategory": "structure"},
            {"id": 2, "name": "column", "supercategory": "structure"},
            {"id": 3, "name": "cell", "supercategory": "structure"}
        ]
    }

    annotation_id = 1

    print(f"Creating {num_images} sample table images...")

    for i in range(num_images):
        # Randomize table structure
        rows = random.randint(2, 6)
        cols = random.randint(2, 5)
        width = random.randint(600, 1000)
        height = random.randint(400, 800)

        # Create image and annotations
        image, annotations = create_sample_table_image(width, height, rows, cols)

        # Save image
        image_filename = f"sample_table_{i+1:03d}.jpg"
        image_path = output_path / image_filename
        image.save(image_path, "JPEG", quality=95)

        # Add image info
        image_info = {
            "id": i + 1,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "date_captured": "2025-09-21"
        }
        dataset_info["images"].append(image_info)

        # Add annotations
        for ann in annotations:
            ann_info = {
                "id": annotation_id,
                "image_id": i + 1,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": 0
            }
            dataset_info["annotations"].append(ann_info)
            annotation_id += 1

        if (i + 1) % 5 == 0:
            print(f"Created {i + 1}/{num_images} images")

    # Save annotations
    annotations_file = output_path / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"✅ Sample dataset created!")
    print(f"📁 Location: {output_path}")
    print(f"📊 Statistics:")
    print(f"   • Images: {len(dataset_info['images'])}")
    print(f"   • Annotations: {len(dataset_info['annotations'])}")
    print(f"   • Categories: {len(dataset_info['categories'])}")

    return str(output_path)

def main():
    """Create sample dataset"""
    dataset_path = create_sample_dataset(num_images=50)  # Create 50 sample images

    print(f"\n🎯 Next steps:")
    print(f"1. Run: python scripts/data_converter.py")
    print(f"2. Run: python train_table_detection.py")

if __name__ == "__main__":
    main()