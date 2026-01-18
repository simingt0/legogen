"""
YOLO training script for LEGO brick detection.
Supports multiple YOLO versions and automatic model selection.
"""

import os
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime
import wandb
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class YOLOTrainer:
    """YOLO trainer for LEGO brick detection."""
    
    def __init__(self, data_yaml: str, model_size: str = "m", yolo_version: str = "v8"):
        """
        Initialize trainer.
        
        Args:
            data_yaml: Path to YOLO dataset YAML file
            model_size: Model size (n, s, m, l, x)
            yolo_version: YOLO version (v5, v8, v9)
        """
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size
        self.yolo_version = yolo_version
        self.model = None
        self.results = None
        
        # Create output directory
        self.output_dir = Path(f"models/yolo{self.yolo_version}_{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset info
        with open(self.data_yaml, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
        
        self.num_classes = self.dataset_info['nc']
        self.class_names = self.dataset_info['names']
    
    def get_model_name(self) -> str:
        """Get the YOLO model name."""
        if self.yolo_version == "v8":
            return f"yolov8{self.model_size}.pt"
        elif self.yolo_version == "v9":
            return f"yolov9{self.model_size}.pt"
        elif self.yolo_version == "v5":
            return f"yolov5{self.model_size}.pt"
        else:
            raise ValueError(f"Unsupported YOLO version: {self.yolo_version}")
    
    def load_model(self, pretrained: bool = True):
        """Load YOLO model."""
        print(f"Loading YOLO{self.yolo_version} model ({self.model_size} size)...")
        
        model_name = self.get_model_name()
        
        try:
            if pretrained:
                self.model = YOLO(model_name)
                print(f"Loaded pretrained model: {model_name}")
            else:
                self.model = YOLO(f"yolov8{self.model_size}.yaml")  # Create from scratch
                print(f"Created model from scratch: {model_name}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to download model...")
            self.model = YOLO(model_name)
    
    def train(self, epochs: int = 100, batch_size: int = 16, learning_rate: float = 0.01,
              img_size: int = 640, patience: int = 50, save_period: int = 10,
              use_wandb: bool = True, device: str = None):
        """Train the model."""
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Training on device: {device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="lego-brick-detection",
                name=f"yolo{self.yolo_version}_{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": f"yolo{self.yolo_version}_{self.model_size}",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "img_size": img_size,
                    "num_classes": self.num_classes
                }
            )
        
        # Training arguments
        train_args = {
            'data': str(self.data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': learning_rate,
            'patience': patience,
            'save_period': save_period,
            'project': str(self.output_dir),
            'name': 'train',
            'device': device,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Class loss gain
            'dfl': 1.5,  # DFL loss gain
            'hsv_h': 0.015,  # HSV Hue augmentation
            'hsv_s': 0.7,   # HSV Saturation augmentation
            'hsv_v': 0.4,   # HSV Value augmentation
            'degrees': 0.0,  # Rotation augmentation
            'translate': 0.1,  # Translation augmentation
            'scale': 0.5,  # Scale augmentation
            'shear': 0.0,  # Shear augmentation
            'perspective': 0.0,  # Perspective augmentation
            'flipud': 0.0,  # Vertical flip
            'fliplr': 0.5,  # Horizontal flip
            'mosaic': 1.0,  # Mosaic augmentation
            'mixup': 0.0,   # Mixup augmentation
            'copy_paste': 0.0  # Copy-paste augmentation
        }
        
        print("Starting training...")
        self.results = self.model.train(**train_args)
        
        # Save training results
        self.save_training_results()
        
        if use_wandb:
            wandb.finish()
        
        print(f"Training completed! Results saved to: {self.output_dir}")
    
    def validate(self):
        """Validate the trained model."""
        print("Validating model...")
        
        # Run validation
        metrics = self.model.val()
        
        # Save validation results
        val_results = {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'class_metrics': {}
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            val_results['class_metrics'][class_name] = {
                'precision': metrics.box.p[i],
                'recall': metrics.box.r[i],
                'map50': metrics.box.ap50[i],
                'map': metrics.box.ap[i]
            }
        
        # Save results
        with open(self.output_dir / 'validation_results.json', 'w') as f:
            import json
            json.dump(val_results, f, indent=2)
        
        print("Validation completed!")
        print(f"mAP50: {val_results['mAP50']:.4f}")
        print(f"mAP50-95: {val_results['mAP50-95']:.4f}")
        
        return val_results
    
    def save_training_results(self):
        """Save training results and plots."""
        if self.results is None:
            return
        
        # Save model
        model_path = self.output_dir / 'best_model.pt'
        self.model.save(str(model_path))
        print(f"Best model saved to: {model_path}")
        
        # Save training plots if available
        try:
            # Plot training results
            self.plot_training_curves()
        except Exception as e:
            print(f"Could not create training plots: {e}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        try:
            # Get training results
            results_csv = list(self.output_dir.glob("**/results.csv"))
            if results_csv:
                import pandas as pd
                results_df = pd.read_csv(results_csv[0])
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('Training Curves')
                
                # Plot losses
                axes[0, 0].plot(results_df['epoch'], results_df['train/box_loss'], label='Box Loss')
                axes[0, 0].plot(results_df['epoch'], results_df['train/cls_loss'], label='Class Loss')
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                
                # Plot metrics
                axes[0, 1].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP50')
                axes[0, 1].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP50-95')
                axes[0, 1].set_title('Validation Metrics')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                
                # Plot precision and recall
                axes[1, 0].plot(results_df['epoch'], results_df['metrics/precision(B)'], label='Precision')
                axes[1, 0].plot(results_df['epoch'], results_df['metrics/recall(B)'], label='Recall')
                axes[1, 0].set_title('Precision and Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                
                # Plot learning rate
                axes[1, 1].plot(results_df['epoch'], results_df['lr/pg0'], label='LR pg0')
                axes[1, 1].plot(results_df['epoch'], results_df['lr/pg1'], label='LR pg1')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Could not plot training curves: {e}")
    
    def export_model(self, format: str = 'onnx'):
        """Export model to different formats."""
        print(f"Exporting model to {format} format...")
        
        try:
            # Export model
            exported_model = self.model.export(format=format)
            print(f"Model exported to: {exported_model}")
            return exported_model
        except Exception as e:
            print(f"Error exporting model: {e}")
            return None

def compare_models(data_yaml: str, model_sizes: list = None, yolo_versions: list = None):
    """Compare different YOLO models and configurations."""
    
    if model_sizes is None:
        model_sizes = ['n', 's', 'm']
    
    if yolo_versions is None:
        yolo_versions = ['v8']
    
    results = {}
    
    for version in yolo_versions:
        for size in model_sizes:
            print(f"\n{'='*50}")
            print(f"Training YOLO{version} {size}")
            print(f"{'='*50}")
            
            try:
                trainer = YOLOTrainer(data_yaml, size, version)
                trainer.load_model()
                trainer.train(epochs=50)  # Shorter training for comparison
                val_results = trainer.validate()
                
                results[f"yolo{version}_{size}"] = {
                    'mAP50': val_results['mAP50'],
                    'mAP50-95': val_results['mAP50-95'],
                    'precision': val_results['precision'],
                    'recall': val_results['recall']
                }
                
            except Exception as e:
                print(f"Error training YOLO{version} {size}: {e}")
                results[f"yolo{version}_{size}"] = None
    
    # Print comparison results
    print(f"\n{'='*50}")
    print("Model Comparison Results")
    print(f"{'='*50}")
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name}:")
            print(f"  mAP50: {metrics['mAP50']:.4f}")
            print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for LEGO brick detection")
    parser.add_argument("--data-yaml", type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument("--model-size", type=str, default="m", choices=['n', 's', 'm', 'l', 'x'],
                       help="Model size")
    parser.add_argument("--yolo-version", type=str, default="v8", choices=['v5', 'v8', 'v9'],
                       help="YOLO version")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--export-format", type=str, default=None, choices=['onnx', 'torchscript', 'tensorrt'],
                       help="Export model format after training")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        compare_models(args.data_yaml)
    else:
        # Train single model
        trainer = YOLOTrainer(args.data_yaml, args.model_size, args.yolo_version)
        trainer.load_model()
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            img_size=args.img_size,
            use_wandb=not args.no_wandb,
            device=args.device
        )
        
        # Validate
        trainer.validate()
        
        # Export if requested
        if args.export_format:
            trainer.export_model(args.export_format)

if __name__ == "__main__":
    main()