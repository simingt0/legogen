"""
Dataset preparation script for LEGO brick detection dataset from Kaggle.
Handles downloading, organizing, and preparing data for YOLO training.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class LegoDatasetPreparer:
    """Prepares LEGO brick detection dataset for YOLO training."""
    
    def __init__(self, data_dir: str = "data", kaggle_dataset: str = "ronanpickell/b100-lego-detection-dataset"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.kaggle_dataset = kaggle_dataset
        
        # YOLO format directories
        self.yolo_dir = self.data_dir / "yolo"
        self.images_dir = self.yolo_dir / "images"
        self.labels_dir = self.yolo_dir / "labels"
        
        # Create directories
        for dir_path in [self.raw_data_dir, self.processed_dir, self.yolo_dir, 
                        self.images_dir / "train", self.images_dir / "val", 
                        self.labels_dir / "train", self.labels_dir / "val"]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, kaggle_username: str = None, kaggle_key: str = None):
        """Download dataset from Kaggle."""
        print("Downloading Kaggle dataset...")
        
        # Set Kaggle credentials if provided
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
        
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                self.kaggle_dataset,
                path=str(self.raw_data_dir),
                unzip=True
            )
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/ronanpickell/b100-lego-detection-dataset")
            return False
        return True
    
    def parse_annotations(self, annotation_path: str) -> List[Dict]:
        """Parse annotation file and extract bounding box information."""
        annotations = []
        
        # Assuming annotations are in COCO, YOLO, or Pascal VOC format
        # This needs to be adapted based on the actual format of the Kaggle dataset
        if annotation_path.endswith('.json'):
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                
            # Parse COCO format
            if 'annotations' in data and 'images' in data:
                image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
                
                for ann in data['annotations']:
                    if 'bbox' in ann:
                        annotations.append({
                            'image_file': image_id_to_filename[ann['image_id']],
                            'bbox': ann['bbox'],  # [x, y, width, height]
                            'category_id': ann['category_id'],
                            'category_name': self.get_category_name(data, ann['category_id'])
                        })
        
        return annotations
    
    def get_category_name(self, coco_data: Dict, category_id: int) -> str:
        """Get category name from COCO data."""
        for cat in coco_data.get('categories', []):
            if cat['id'] == category_id:
                return cat['name']
        return 'unknown'
    
    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert bounding box to YOLO format."""
        x, y, width, height = bbox
        
        # Convert to center coordinates
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        
        # Normalize width and height
        width_norm = width / img_width
        height_norm = height / img_height
        
        return [x_center, y_center, width_norm, height_norm]
    
    def create_class_mapping(self, annotations: List[Dict]) -> Dict[str, int]:
        """Create class name to ID mapping."""
        unique_classes = set(ann['category_name'] for ann in annotations)
        return {class_name: idx for idx, class_name in enumerate(sorted(unique_classes))}
    
    def process_dataset(self, test_size: float = 0.2, random_state: int = 42):
        """Process the raw dataset and convert to YOLO format."""
        print("Processing dataset...")
        
        # Find annotation files
        annotation_files = list(self.raw_data_dir.rglob("*.json"))
        if not annotation_files:
            print("No annotation files found. Please check dataset structure.")
            return False
        
        all_annotations = []
        for ann_file in annotation_files:
            annotations = self.parse_annotations(str(ann_file))
            all_annotations.extend(annotations)
        
        if not all_annotations:
            print("No annotations found.")
            return False
        
        # Create class mapping
        class_mapping = self.create_class_mapping(all_annotations)
        print(f"Found {len(class_mapping)} classes: {list(class_mapping.keys())}")
        
        # Group annotations by image
        image_annotations = {}
        for ann in all_annotations:
            img_file = ann['image_file']
            if img_file not in image_annotations:
                image_annotations[img_file] = []
            image_annotations[img_file].append(ann)
        
        # Split into train/val
        image_files = list(image_annotations.keys())
        train_files, val_files = train_test_split(
            image_files, test_size=test_size, random_state=random_state
        )
        
        print(f"Train images: {len(train_files)}, Val images: {len(val_files)}")
        
        # Process train and validation sets
        for split, files in [('train', train_files), ('val', val_files)]:
            self.process_split(split, files, image_annotations, class_mapping)
        
        # Create dataset configuration
        self.create_dataset_yaml(class_mapping)
        
        return True
    
    def process_split(self, split: str, image_files: List[str], 
                       image_annotations: Dict, class_mapping: Dict[str, int]):
        """Process a train/validation split."""
        split_images_dir = self.images_dir / split
        split_labels_dir = self.labels_dir / split
        
        for img_file in tqdm(image_files, desc=f"Processing {split} split"):
            # Find image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = self.raw_data_dir / img_file
                if potential_path.exists():
                    img_path = potential_path
                    break
                # Try without extension
                potential_path = self.raw_data_dir / img_file.replace('.jpg', ext).replace('.png', ext)
                if potential_path.exists():
                    img_path = potential_path
                    break
            
            if not img_path or not img_path.exists():
                print(f"Image not found: {img_file}")
                continue
            
            # Copy image to YOLO directory
            dst_img_path = split_images_dir / img_path.name
            shutil.copy2(img_path, dst_img_path)
            
            # Load image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not load image: {img_path}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Create YOLO label file
            label_file = split_labels_dir / f"{img_path.stem}.txt"
            
            with open(label_file, 'w') as f:
                for ann in image_annotations.get(img_file, []):
                    class_id = class_mapping[ann['category_name']]
                    yolo_bbox = self.convert_bbox_to_yolo(
                        ann['bbox'], img_width, img_height
                    )
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
    
    def create_dataset_yaml(self, class_mapping: Dict[str, int]):
        """Create YOLO dataset configuration file."""
        dataset_config = {
            'path': str(self.yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_mapping),
            'names': {v: k for k, v in class_mapping.items()}
        }
        
        config_path = self.yolo_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {config_path}")
    
    def visualize_dataset(self, num_samples: int = 5):
        """Visualize some samples from the dataset."""
        import matplotlib.pyplot as plt
        import random
        
        # Load a few random images
        train_images = list((self.images_dir / "train").glob("*.jpg"))
        if not train_images:
            train_images = list((self.images_dir / "train").glob("*.png"))
        
        if not train_images:
            print("No images found for visualization")
            return
        
        samples = random.sample(train_images, min(num_samples, len(train_images)))
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
        
        for i, img_path in enumerate(samples):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load corresponding labels
            label_path = self.labels_dir / "train" / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    labels = f.readlines()
                
                # Draw bounding boxes
                h, w = img.shape[:2]
                for label in labels:
                    parts = label.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Convert back to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Sample {i+1}")
        
        plt.tight_layout()
        plt.savefig(self.processed_dir / 'dataset_samples.png')
        print(f"Visualization saved to: {self.processed_dir / 'dataset_samples.png'}")

def main():
    parser = argparse.ArgumentParser(description="Prepare LEGO brick detection dataset for YOLO training")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--kaggle-username", type=str, help="Kaggle username")
    parser.add_argument("--kaggle-key", type=str, help="Kaggle API key")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--visualize", action="store_true", help="Visualize dataset samples")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    
    args = parser.parse_args()
    
    preparer = LegoDatasetPreparer(args.data_dir)
    
    # Download dataset
    if not args.skip_download:
        success = preparer.download_kaggle_dataset(args.kaggle_username, args.kaggle_key)
        if not success:
            print("Dataset download failed. Exiting.")
            return
    
    # Process dataset
    success = preparer.process_dataset(test_size=args.test_size)
    if not success:
        print("Dataset processing failed.")
        return
    
    # Visualize if requested
    if args.visualize:
        preparer.visualize_dataset()
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main()