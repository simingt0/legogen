"""
Brickognize API integration for LEGO brick identification.
Handles API calls, image processing, and result parsing.
"""

import os
import json
import base64
import asyncio
import aiohttp
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import io
import time
from dataclasses import dataclass
from enum import Enum

class BrickType(Enum):
    """LEGO brick types supported by Brickognize API."""
    BRICK = "brick"
    PLATE = "plate"
    TILE = "tile"
    SLOPE = "slope"
    ARCH = "arch"
    TECHNIC = "technic"
    OTHER = "other"

@dataclass
class BrickPrediction:
    """Brick prediction result."""
    part_number: str
    name: str
    confidence: float
    brick_type: BrickType
    color: Optional[str] = None
    dimensions: Optional[Tuple[int, int, int]] = None
    image_url: Optional[str] = None

class BrickognizeClient:
    """Client for Brickognize API."""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.brickognize.com"):
        """
        Initialize client.
        
        Args:
            api_key: API key for Brickognize (optional for free tier)
            base_url: Base URL for the API
        """
        self.api_key = api_key or os.getenv("BRICKOGNIZE_API_KEY")
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        headers = {
            "User-Agent": "LegoGen-Brickognize-Client/1.0",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        self.session = aiohttp.ClientSession(headers=headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _get_sync_headers(self) -> Dict:
        """Gets headers for synchronous requests."""
        headers = {
            "User-Agent": "LegoGen-Brickognize-Client/1.0",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def identify_brick_from_detection(self, original_image: np.ndarray, detection_bbox: List[float]) -> Dict:
        """
        Identify LEGO brick from a detected bounding box (SYNC).
        
        Args:
            original_image: Original image as numpy array
            detection_bbox: Bounding box [x1, y1, x2, y2] from detection
            
        Returns:
            API response with brick identification results
        """
        pil_image = self._prepare_image_from_detection(original_image, detection_bbox)
        image_bytes = self.image_to_bytes(pil_image)
        return self.identify_brick(image_bytes)

    async def identify_brick_from_detection_async(self, original_image: np.ndarray, detection_bbox: List[float]) -> Dict:
        """
        Identify LEGO brick from a detected bounding box (ASYNC).
        
        Args:
            original_image: Original image as numpy array
            detection_bbox: Bounding box [x1, y1, x2, y2] from detection
            
        Returns:
            API response with brick identification results
        """
        pil_image = self._prepare_image_from_detection(original_image, detection_bbox)
        image_bytes = self.image_to_bytes(pil_image)
        return await self.identify_brick_async(image_bytes)

    def _prepare_image_from_detection(self, original_image: np.ndarray, detection_bbox: List[float]) -> Image.Image:
        """Crops, pads, and converts a detected region to a PIL Image."""
        x1, y1, x2, y2 = map(int, detection_bbox)
        cropped_image = original_image[y1:y2, x1:x2]
        padded_image = self.add_padding(cropped_image, padding_ratio=0.1)
        return Image.fromarray(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))

    def identify_brick(self, image_bytes: bytes) -> Dict:
        """
        Identify LEGO brick from image bytes (SYNC).
        
        Args:
            image_bytes: Image file contents as bytes
            
        Returns:
            API response with brick identification results
        """
        url = f"{self.base_url}/predict/"
        files = {'query_image': ('image.jpg', image_bytes, 'image/jpeg')}
        
        try:
            response = requests.post(url, headers=self._get_sync_headers(), files=files, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {"error": str(e), "predictions": []}
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return {"error": "Invalid JSON response", "predictions": []}
    
    async def identify_brick_async(self, image_bytes: bytes) -> Dict:
        """
        Identify LEGO brick from image bytes (ASYNC).
        
        Args:
            image_bytes: Image file contents as bytes
            
        Returns:
            API response with brick identification results
        """
        url = f"{self.base_url}/predict/"
        
        data = aiohttp.FormData()
        data.add_field('query_image',
                       image_bytes,
                       filename='image.jpg',
                       content_type='image/jpeg')
        
        try:
            print(f"Sending request to {url}...")
            async with self.session.post(url, data=data) as response:
                response_text = await response.text()
                print(f"Received response (status {response.status}): {response_text}")
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Async API request failed: {e}")
            return {"error": str(e), "predictions": []}
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from response: {response_text}")
            return {"error": "Invalid JSON response", "predictions": []}
    
    def load_and_preprocess_image(self, image_path: str, crop_coordinates: Tuple[int, int, int, int] = None) -> Image.Image:
        """Load and preprocess image."""
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply crop if provided
        if crop_coordinates:
            x1, y1, x2, y2 = crop_coordinates
            image = image.crop((x1, y1, x2, y2))
        
        # Resize if too large (API might have size limits)
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
    
    def preprocess_image_array(self, image_array: np.ndarray, crop_coordinates: Tuple[int, int, int, int] = None) -> Image.Image:
        """Preprocess numpy array image."""
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        
        # Apply crop if provided
        if crop_coordinates:
            x1, y1, x2, y2 = crop_coordinates
            image = image.crop((x1, y1, x2, y2))
        
        # Resize if too large
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
    
    def add_padding(self, image: np.ndarray, padding_ratio: float = 0.1) -> np.ndarray:
        """Add padding around the image."""
        h, w = image.shape[:2]
        pad_h = int(h * padding_ratio)
        pad_w = int(w * padding_ratio)
        
        # Create padded image with white background
        padded = np.ones((h + 2 * pad_h, w + 2 * pad_w, 3), dtype=np.uint8) * 255
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
        
        return padded
    
    def image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return buffered.getvalue()
    
    def parse_predictions(self, api_response: Dict) -> List[BrickPrediction]:
        """Parse API response into BrickPrediction objects."""
        predictions = []
        
        if 'error' in api_response:
            print(f"API Error: {api_response['error']}")
            return predictions
        
        for pred in api_response.get('items', []):
            try:
                category_str = pred.get('category', 'other').lower()
                brick_type = BrickType.OTHER
                if 'brick' in category_str:
                    brick_type = BrickType.BRICK
                elif 'plate' in category_str:
                    brick_type = BrickType.PLATE
                elif 'tile' in category_str:
                    brick_type = BrickType.TILE
                elif 'slope' in category_str:
                    brick_type = BrickType.SLOPE
                elif 'arch' in category_str:
                    brick_type = BrickType.ARCH
                elif 'technic' in category_str:
                    brick_type = BrickType.TECHNIC

                brick_pred = BrickPrediction(
                    part_number=pred.get('id', 'unknown'),
                    name=pred.get('name', 'Unknown Brick'),
                    confidence=float(pred.get('score', 0.0)),
                    brick_type=brick_type,
                    color=pred.get('color'),
                    dimensions=self.parse_dimensions(pred.get('dimensions')),
                    image_url=pred.get('img_url')
                )
                predictions.append(brick_pred)
                
            except (ValueError, TypeError) as e:
                print(f"Error parsing prediction score: {e} - Prediction: {pred}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred during prediction parsing: {e}")
                continue
        
        return predictions
    
    def parse_dimensions(self, dimensions_str: str) -> Optional[Tuple[int, int, int]]:
        """Parse dimensions string into tuple."""
        if not dimensions_str:
            return None
        
        try:
            # Assuming format like "2x4x1" or "2x4"
            parts = dimensions_str.split('x')
            if len(parts) >= 2:
                width = int(parts[0])
                length = int(parts[1])
                height = int(parts[2]) if len(parts) > 2 else 1
                return (width, length, height)
        except (ValueError, IndexError):
            pass
        
        return None
    
    def batch_identify_bricks(self, image_paths: List[str], crop_coordinates_list: List[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """
        Batch identify multiple bricks.
        
        Args:
            image_paths: List of image paths
            crop_coordinates_list: Optional list of crop coordinates
            
        Returns:
            List of API responses
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            crop_coords = crop_coordinates_list[i] if crop_coordinates_list else None
            
            try:
                result = self.identify_brick_from_file(image_path, crop_coords)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({"error": str(e), "predictions": []})
            
            # Add small delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    async def batch_identify_bricks_async(self, image_paths: List[str], crop_coordinates_list: List[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """Async batch identify multiple bricks."""
        async with self:
            tasks = []
            
            for i, image_path in enumerate(image_paths):
                crop_coords = crop_coordinates_list[i] if crop_coordinates_list else None
                
                # Preprocess image
                image = self.load_and_preprocess_image(image_path, crop_coords)
                image_base64 = self.image_to_base64(image)
                
                # Create async task
                task = self.identify_brick_async(image_base64)
                tasks.append(task)
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

class BrickognizePipeline:
    """Pipeline for processing YOLO detections with Brickognize API."""
    
    def __init__(self, api_key: str = None, confidence_threshold: float = 0.7, concurrency_limit: int = 5):
        """
        Initialize pipeline.
        
        Args:
            api_key: Brickognize API key
            confidence_threshold: Minimum confidence for considering predictions
            concurrency_limit: Max number of concurrent requests to the API
        """
        self.client = BrickognizeClient(api_key)
        self.confidence_threshold = confidence_threshold
        self.semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def _process_single_detection(self, client, original_image, detection):
        """Helper to process a single detection with semaphore control."""
        async with self.semaphore:
            await asyncio.sleep(1)  # Add a small delay to avoid rate-limiting
            return await client.identify_brick_from_detection_async(original_image, detection['box'])

    async def process_detections(self, original_image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Process YOLO detections through Brickognize API concurrently with rate limiting.
        
        Args:
            original_image: Original image as numpy array
            detections: List of YOLO detection results from the pipeline
            
        Returns:
            List of enhanced detection results with Brickognize predictions
        """
        enhanced_detections = []
        
        async with self.client as client:
            tasks = [self._process_single_detection(client, original_image, det) for det in detections]
            brick_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(brick_results):
            det = detections[i]
            box = det['box']
            confidence = det['confidence']

            # Default placeholder for the prediction
            best_prediction_data = {
                'part_number': 'unknown',
                'name': 'Unknown Piece',
                'confidence': 0.0,
            }

            if isinstance(result, Exception):
                print(f"Error processing detection {i}: {result}")
            else:
                brick_predictions = self.client.parse_predictions(result)
                valid_predictions = [p for p in brick_predictions if p.confidence >= self.confidence_threshold]
                
                if valid_predictions:
                    best_prediction = max(valid_predictions, key=lambda p: p.confidence)
                    best_prediction_data['part_number'] = best_prediction.part_number
                    best_prediction_data['name'] = best_prediction.name
                    best_prediction_data['confidence'] = best_prediction.confidence

            enhanced_detection = {
                'yolo_bbox': box,
                'yolo_confidence': confidence,
                'brickognize_prediction': best_prediction_data
            }
            enhanced_detections.append(enhanced_detection)
                
        return enhanced_detections
    
    def create_brick_inventory(self, enhanced_detections: List[Dict]) -> Dict:
        """
        Create brick inventory from enhanced detections.
        
        Args:
            enhanced_detections: List of enhanced detection results
            
        Returns:
            Brick inventory with counts and details
        """
        inventory = {}
        
        for detection in enhanced_detections:
            brick_pred = detection['brickognize_prediction']
            part_number = brick_pred['part_number']
            
            if part_number not in inventory:
                inventory[part_number] = {
                    'name': brick_pred['name'],
                    'brick_type': brick_pred['brick_type'],
                    'color': brick_pred['color'],
                    'dimensions': brick_pred['dimensions'],
                    'count': 0,
                    'detections': []
                }
            
            inventory[part_number]['count'] += 1
            inventory[part_number]['detections'].append({
                'yolo_confidence': detection['yolo_confidence'],
                'brickognize_confidence': brick_pred['confidence'],
                'bbox': detection['yolo_bbox']
            })
        
        return inventory

def main():
    """Example usage of Brickognize API integration."""

    # Initialize client
    api_key = os.getenv("BRICKOGNIZE_API_KEY")
    if not api_key:
        print("Please set BRICKOGNIZE_API_KEY environment variable")
        return
    
    client = BrickognizeClient(api_key)
    
    # Example 1: Identify brick from file
    print("Example 1: Identifying brick from file...")
    result = client.identify_brick_from_file("path/to/brick_image.jpg")
    predictions = client.parse_predictions(result)
    
    for i, pred in enumerate(predictions[:3]):  # Show top 3 predictions
        print(f"Prediction {i+1}:")
        print(f"  Part Number: {pred.part_number}")
        print(f"  Name: {pred.name}")
        print(f"  Confidence: {pred.confidence:.2f}")
        print(f"  Type: {pred.brick_type.value}")
        if pred.dimensions:
            print(f"  Dimensions: {pred.dimensions[0]}x{pred.dimensions[1]}x{pred.dimensions[2]}")
        print()
    
    # Example 2: Process YOLO detections
    print("Example 2: Processing YOLO detections...")
    pipeline = BrickognizePipeline(api_key)
    
    # Mock YOLO detections (in real usage, these would come from YOLO model)
    mock_detections = [
        {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.85,
            'class': 'brick'
        },
        {
            'bbox': [250, 150, 350, 250],
            'confidence': 0.92,
            'class': 'plate'
        }
    ]
    
    # Mock image (in real usage, this would be the actual image)
    mock_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    enhanced_detections = pipeline.process_yolo_detections(mock_image, mock_detections)
    inventory = pipeline.create_brick_inventory(enhanced_detections)
    
    print("Brick Inventory:")
    for part_number, details in inventory.items():
        print(f"  {part_number}: {details['name']} x{details['count']}")

if __name__ == "__main__":
    main()