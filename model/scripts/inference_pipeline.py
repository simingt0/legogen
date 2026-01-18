"""
Complete inference pipeline for LEGO brick detection and identification.
Combines YOLO detection with Brickognize API for enhanced identification.
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

# Import our custom modules
from brickognize.brickognize_client import BrickognizeClient, BrickognizePipeline

class LegoInferencePipeline:
    """Complete inference pipeline for LEGO brick detection and identification."""
    
    def __init__(self, yolo_model_path: str, brickognize_api_key: str = None, 
                 confidence_threshold: float = 0.5, use_brickognize: bool = True):
        """
        Initialize inference pipeline.
        
        Args:
            yolo_model_path: Path to trained YOLO model
            brickognize_api_key: API key for Brickognize (optional)
            confidence_threshold: Minimum confidence for YOLO detections
            use_brickognize: Whether to use Brickognize API for enhanced identification
        """
        self.yolo_model_path = Path(yolo_model_path)
        self.confidence_threshold = confidence_threshold
        self.use_brickognize = use_brickognize
        
        # Load YOLO model
        print(f"Loading YOLO model from: {self.yolo_model_path}")
        self.yolo_model = YOLO(str(self.yolo_model_path))
        
        # Initialize Brickognize if enabled
        if self.use_brickognize and brickognize_api_key:
            print("Initializing Brickognize integration...")
            self.brickognize_pipeline = BrickognizePipeline(brickognize_api_key)
        else:
            self.brickognize_pipeline = None
            if self.use_brickognize:
                print("Warning: Brickognize API key not provided. Using YOLO-only detection.")
        
        # Results storage
        self.inference_results = []
    
    def detect_bricks(self, image: np.ndarray, return_image: bool = False) -> Dict:
        """
        Detect LEGO bricks in image using YOLO.
        
        Args:
            image: Input image as numpy array
            return_image: Whether to return annotated image
            
        Returns:
            Detection results with bounding boxes and confidence scores
        """
        # Run YOLO inference
        results = self.yolo_model(image, conf=self.confidence_threshold)
        
        detections = []
        annotated_image = None
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extract detections
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                
                for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                    if score >= self.confidence_threshold:
                        detection = {
                            'bbox': box.tolist(),  # [x1, y1, x2, y2]
                            'class_id': int(cls),
                            'class_name': self.yolo_model.names[cls] if hasattr(self.yolo_model, 'names') else f'class_{cls}',
                            'confidence': float(score),
                            'area': float((box[2] - box[0]) * (box[3] - box[1]))
                        }
                        detections.append(detection)
            
            # Get annotated image if requested
            if return_image:
                annotated_image = result.plot()
        
        # Sort detections by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        results_dict = {
            'num_detections': len(detections),
            'detections': detections,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_image:
            results_dict['annotated_image'] = annotated_image
        
        return results_dict
    
    def identify_bricks(self, image: np.ndarray, yolo_detections: List[Dict]) -> List[Dict]:
        """
        Identify specific brick types using Brickognize API.
        
        Args:
            image: Original image as numpy array
            yolo_detections: YOLO detection results
            
        Returns:
            Enhanced detections with Brickognize identification
        """
        if not self.brickognize_pipeline:
            print("Brickognize pipeline not available. Returning YOLO detections only.")
            return yolo_detections
        
        try:
            # Process detections through Brickognize
            enhanced_detections = self.brickognize_pipeline.process_yolo_detections(
                image, yolo_detections
            )
            
            return enhanced_detections
            
        except Exception as e:
            print(f"Error in Brickognize identification: {e}")
            return yolo_detections
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     save_results: bool = True, visualize: bool = True) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (optional)
            save_results: Whether to save results to files
            visualize: Whether to create visualizations
            
        Returns:
            Complete inference results
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: YOLO detection
        print("Running YOLO detection...")
        detection_results = self.detect_bricks(image_rgb, return_image=True)
        
        # Step 2: Brickognize identification (if enabled)
        if self.use_brickognize and self.brickognize_pipeline:
            print("Running Brickognize identification...")
            enhanced_detections = self.identify_bricks(image_rgb, detection_results['detections'])
            
            # Create brick inventory
            inventory = self.brickognize_pipeline.create_brick_inventory(enhanced_detections)
            
            # Update results
            detection_results['enhanced_detections'] = enhanced_detections
            detection_results['brick_inventory'] = inventory
        
        # Add image info
        detection_results['image_path'] = image_path
        detection_results['image_shape'] = image.shape
        
        # Save results
        if save_results and output_dir:
            self.save_inference_results(detection_results, output_dir, visualize)
        
        # Store in memory
        self.inference_results.append(detection_results)
        
        return detection_results
    
    def process_batch(self, image_paths: List[str], output_dir: str, 
                     save_results: bool = True, visualize: bool = True) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            save_results: Whether to save results to files
            visualize: Whether to create visualizations
            
        Returns:
            List of inference results for all images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.process_image(
                    image_path, 
                    str(output_dir / f"image_{i+1}"),
                    save_results,
                    visualize
                )
                batch_results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                batch_results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'num_detections': 0,
                    'detections': []
                })
        
        # Save batch summary
        if save_results:
            self.save_batch_summary(batch_results, output_dir)
        
        return batch_results
    
    def save_inference_results(self, results: Dict, output_dir: str, visualize: bool = True):
        """Save inference results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_file = output_dir / "inference_results.json"
        with open(json_file, 'w') as f:
            # Remove image data for JSON serialization
            json_results = results.copy()
            if 'annotated_image' in json_results:
                del json_results['annotated_image']
            json.dump(json_results, f, indent=2)
        
        # Save annotated image
        if 'annotated_image' in results:
            image_file = output_dir / "annotated_image.jpg"
            annotated_image = results['annotated_image']
            if isinstance(annotated_image, np.ndarray):
                cv2.imwrite(str(image_file), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(image_file), annotated_image)
        
        # Save brick inventory if available
        if 'brick_inventory' in results:
            inventory_file = output_dir / "brick_inventory.json"
            with open(inventory_file, 'w') as f:
                json.dump(results['brick_inventory'], f, indent=2)
        
        # Create visualizations
        if visualize:
            self.create_visualizations(results, output_dir)
        
        print(f"Results saved to: {output_dir}")
    
    def create_visualizations(self, results: Dict, output_dir: Path):
        """Create visualization plots."""
        # Detection confidence histogram
        if results['detections']:
            confidences = [d['confidence'] for d in results['detections']]
            
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.hist(confidences, bins=20, edgecolor='black')
            plt.title('Detection Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            
            # Class distribution
            classes = [d['class_name'] for d in results['detections']]
            unique_classes = list(set(classes))
            class_counts = [classes.count(c) for c in unique_classes]
            
            plt.subplot(2, 2, 2)
            plt.bar(unique_classes, class_counts)
            plt.title('Detected Classes')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Detection areas
            areas = [d['area'] for d in results['detections']]
            
            plt.subplot(2, 2, 3)
            plt.scatter(range(len(areas)), areas, alpha=0.6)
            plt.title('Detection Areas')
            plt.xlabel('Detection Index')
            plt.ylabel('Area (pixels²)')
            
            # Brick inventory if available
            if 'brick_inventory' in results:
                inventory = results['brick_inventory']
                part_numbers = list(inventory.keys())[:10]  # Top 10
                counts = [inventory[pn]['count'] for pn in part_numbers]
                
                plt.subplot(2, 2, 4)
                plt.barh(part_numbers, counts)
                plt.title('Top 10 Brick Types')
                plt.xlabel('Count')
            
            plt.tight_layout()
            plt.savefig(output_dir / "visualizations.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_batch_summary(self, batch_results: List[Dict], output_dir: Path):
        """Save batch processing summary."""
        summary = {
            'total_images': len(batch_results),
            'successful_images': len([r for r in batch_results if 'error' not in r]),
            'total_detections': sum(r.get('num_detections', 0) for r in batch_results),
            'average_detections_per_image': np.mean([r.get('num_detections', 0) for r in batch_results if 'error' not in r]),
            'processing_timestamp': datetime.now().isoformat(),
            'individual_results': batch_results
        }
        
        # Aggregate brick inventory across all images
        if any('brick_inventory' in r for r in batch_results):
            total_inventory = {}
            for result in batch_results:
                if 'brick_inventory' in result:
                    for part_number, details in result['brick_inventory'].items():
                        if part_number not in total_inventory:
                            total_inventory[part_number] = {
                                'name': details['name'],
                                'brick_type': details['brick_type'],
                                'color': details['color'],
                                'dimensions': details['dimensions'],
                                'count': 0
                            }
                        total_inventory[part_number]['count'] += details['count']
            
            summary['total_brick_inventory'] = total_inventory
        
        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Batch summary saved to: {summary_file}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            'yolo_model_path': str(self.yolo_model_path),
            'confidence_threshold': self.confidence_threshold,
            'use_brickognize': self.use_brickognize,
            'brickognize_available': self.brickognize_pipeline is not None,
            'model_classes': getattr(self.yolo_model, 'names', {}),
            'num_classes': len(getattr(self.yolo_model, 'names', {}))
        }
        
        return info

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="LEGO Brick Detection and Identification Pipeline")
    parser.add_argument("--yolo-model", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--brickognize-api-key", type=str, help="Brickognize API key")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--no-brickognize", action="store_true", help="Disable Brickognize API")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--save-results", action="store_true", help="Save results to files")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LegoInferencePipeline(
        yolo_model_path=args.yolo_model,
        brickognize_api_key=args.brickognize_api_key,
        confidence_threshold=args.confidence_threshold,
        use_brickognize=not args.no_brickognize
    )
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(input_path.glob(ext))
        image_paths = [str(p) for p in image_paths]
    else:
        print(f"Input path not found: {args.input}")
        return
    
    if not image_paths:
        print("No images found in input path")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process images
    if len(image_paths) == 1:
        # Single image
        result = pipeline.process_image(
            image_paths[0], 
            args.output,
            args.save_results,
            args.visualize
        )
        print(f"Processed 1 image, found {result['num_detections']} detections")
    else:
        # Batch processing
        results = pipeline.process_batch(
            image_paths,
            args.output,
            args.save_results,
            args.visualize
        )
        
        total_detections = sum(r.get('num_detections', 0) for r in results)
        print(f"Processed {len(results)} images, found {total_detections} total detections")

if __name__ == "__main__":
    main()