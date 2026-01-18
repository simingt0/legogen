
import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
from brickognize.brickognize_client import BrickognizePipeline

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
        self.model = YOLO(model_path)
        self.brickognize_pipeline = BrickognizePipeline()

    async def run(self, image_path: str) -> list:
        """
        Executes the full inference pipeline on a single image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            list: A list of dictionaries, where each dictionary contains the
                  bounding box and the identification result from Brickognize.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        # 1. Detect all bricks using the YOLO model
        results = self.model(image)
        detections = results[0].boxes.data.cpu().numpy()  # Get bounding boxes

        # 2. Prepare detection data for the Brickognize pipeline
        detection_data = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.5:  # Confidence threshold
                detection_data.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf
                })

        # 3. Process all detected bricks concurrently using Brickognize
        identified_bricks = await self.brickognize_pipeline.process_detections(image, detection_data)

        return identified_bricks

async def main():
    """Main function to run the inference pipeline."""
    # This assumes you have a trained model named 'best.pt' in the project root
    # and an image to test in 'test_image.jpg'.
    # Replace with your actual model and image paths.
    model_path = "c:/Users/jelli/Desktop/legogen/model/runs/legogen_yolo/run1/weights/best.pt"
    image_path = "c:/Users/jelli/Desktop/legogen/model/test_image.jpg"

    pipeline = InferencePipeline(model_path)
    try:
        results = await pipeline.run(image_path)
        print("Identification Results:")
        for result in results:
            print(f"  - Box: {result['box']}, Result: {result['result']}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Create a dummy test image and model path for demonstration if they don't exist
    # In a real scenario, these would be outputs from your training process.
    import os
    if not os.path.exists("c:/Users/jelli/Desktop/legogen/model/runs/legogen_yolo/run1/weights"):
        os.makedirs("c:/Users/jelli/Desktop/legogen/model/runs/legogen_yolo/run1/weights")
    if not os.path.exists("c:/Users/jelli/Desktop/legogen/model/runs/legogen_yolo/run1/weights/best.pt"):
        # Create a dummy file to represent the model
        with open("c:/Users/jelli/Desktop/legogen/model/runs/legogen_yolo/run1/weights/best.pt", "w") as f:
            f.write("dummy yolo model")
    if not os.path.exists(image_path):
        # Create a dummy blank image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy_image)

    asyncio.run(main())
