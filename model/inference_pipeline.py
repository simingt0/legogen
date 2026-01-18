
import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path
from ultralytics import YOLO
# from brickognize.brickognize_client import BrickognizePipeline

class InferencePipeline:
    """
    Coordinates the full LEGO brick identification process, from object detection
    to individual brick classification.
    """
    def __init__(self, model_path: str):
        """
        Initializes the pipeline with a trained YOLO model.

        Args:
            model_path (str): Path to the trained YOLOv8 model file (e.g., 'best.pt').
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        # self.brickognize_pipeline = BrickognizePipeline()

    def run(self, image_path: str) -> list:
        """
        Executes the object detection part of the pipeline on a single image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            list: A list of dictionaries, where each dictionary contains the
                  bounding box and confidence of a detected brick.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        print(f"Processing image: {image_path}")

        # 1. Detect all bricks using the YOLO model
        results = self.model(image, conf=0.25)
        detections = results[0].boxes.data.cpu().numpy()

        print(f"Found {len(detections)} potential bricks.")

        # 2. Prepare detection data
        detection_data = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            detection_data.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf) # Ensure confidence is JSON serializable
            })

        # # 3. Process all detected bricks concurrently using Brickognize
        # print("Identifying bricks with Brickognize API...")
        # identified_bricks = await self.brickognize_pipeline.process_detections(image, detection_data)

        return detection_data

def save_results(output_dir: Path, image: np.ndarray, results: list):
    """Saves the JSON results and the annotated image."""
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON results (detection boxes and confidences)
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {json_path}")

    # 2. Save annotated image
    annotated_image = image.copy()
    for result in results:
        box = result['box']
        confidence = result['confidence']

        # Draw bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"Brick ({confidence:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    image_path = output_dir / "annotated_image.jpg"
    cv2.imwrite(str(image_path), annotated_image)
    print(f"Annotated image saved to: {image_path}")

def main():
    parser = argparse.ArgumentParser(description="Run LEGO brick identification pipeline")
    parser.add_argument("image_path", type=str, help="Path to the image file to analyze")
    parser.add_argument("--model", type=str, 
                        default=r"c:\Users\jelli\Desktop\legogen\model\runs\detect\models\yolov8_s_20260118_004503\train\weights\best.pt",
                        help="Path to the trained YOLO model weights")
    
    args = parser.parse_args()

    try:
        pipeline = InferencePipeline(args.model)
        results = pipeline.run(args.image_path)
        
        # --- Result Processing and Saving ---
        if results:
            image = cv2.imread(args.image_path)
            image_name = Path(args.image_path).stem
            output_dir = Path("runs/predict") / image_name

            save_results(output_dir, image, results)

            print("\n=== Detection Results (Console) ===")
            for i, result in enumerate(results):
                box = result['box']
                confidence = result['confidence']
                print(f"Detection {i+1}: Box={box}, Confidence={confidence:.2f}")
        else:
            print("No detections to save.")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
