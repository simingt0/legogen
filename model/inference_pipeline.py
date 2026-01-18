import argparse
import asyncio
import json
import os
from pathlib import Path

import cv2
import numpy as np
from brickognize.brickognize_client import BrickognizePipeline
from ultralytics import YOLO


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
        self.brickognize_pipeline = BrickognizePipeline()

    async def run(self, image_path: str) -> tuple[list, list, list]:
        """
        Executes the object detection part of the pipeline on a single image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: A tuple containing:
                - list: All identifications from Brickognize (including low confidence).
                - list: High-confidence identifications for final output.
                - list: Raw YOLO detection data.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        print(f"Processing image: {image_path}")

        # 1. Detect all bricks using the YOLO model
        results = self.model(image)
        detections = results[0].boxes.data.cpu().numpy()

        print(f"Found {len(detections)} potential bricks.")

        # 2. Prepare detection data
        detection_data = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            detection_data.append(
                {"box": [int(x1), int(y1), int(x2), int(y2)], "confidence": float(conf)}
            )

        # 3. Process all detected bricks concurrently using Brickognize
        print("Identifying bricks with Brickognize API...")
        (
            all_identifications,
            high_confidence_detections,
        ) = await self.brickognize_pipeline.process_detections(image, detection_data)

        return all_identifications, high_confidence_detections, detection_data


def save_results(
    output_dir: Path,
    image: np.ndarray,
    all_identifications: list,
    high_confidence_results: list,
    yolo_detections: list,
):
    """Saves the JSON results, YOLO image, identified blocks image, and the final annotated image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON results (only high-confidence results)
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(high_confidence_results, f, indent=4)
    print(f"\nResults saved to: {json_path}")

    # 2. Save YOLO-only annotated image
    yolo_image = image.copy()
    for det in yolo_detections:
        box = det["box"]
        conf = det["confidence"]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(yolo_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for YOLO
        label = f"Detection: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(yolo_image, (x1, y1 - h - 5), (x1 + w, y1), (255, 0, 0), -1)
        cv2.putText(
            yolo_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    yolo_image_path = output_dir / "yolo_image.jpg"
    cv2.imwrite(str(yolo_image_path), yolo_image)
    print(f"YOLO detection image saved to: {yolo_image_path}")

    # 3. Save identified blocks image (all identifications including unknown)
    identified_blocks_image = image.copy()
    for result in all_identifications:
        box = result["yolo_bbox"]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(
            identified_blocks_image, (x1, y1), (x2, y2), (255, 255, 0), 2
        )  # Yellow for identified

        pred = result["brickognize_prediction"]
        if pred["confidence"] > 0.0:  # Show all predictions
            label = f"{pred['name']} ({pred['confidence']:.2f})"
        else:
            label = "Unknown"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            identified_blocks_image, (x1, y1 - h - 5), (x1 + w, y1), (255, 255, 0), -1
        )
        cv2.putText(
            identified_blocks_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    identified_blocks_path = output_dir / "identified_blocks.jpg"
    cv2.imwrite(str(identified_blocks_path), identified_blocks_image)
    print(f"Identified blocks image saved to: {identified_blocks_path}")

    # 4. Save final annotated image (only high-confidence results)
    annotated_image = image.copy()
    for result in high_confidence_results:
        box = result["yolo_bbox"]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(
            annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )  # Green for final

        pred = result["brickognize_prediction"]
        label = f"{pred['name']} ({pred['confidence']:.2f})"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    final_image_path = output_dir / "annotated_image.jpg"
    cv2.imwrite(str(final_image_path), annotated_image)
    print(f"Final annotated image saved to: {final_image_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run LEGO brick identification pipeline"
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the image file to analyze"
    )

    # Default model path relative to this script
    script_dir = Path(__file__).parent
    default_model = script_dir / "runs" / "detect" / "train" / "weights" / "best.pt"

    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model),
        help="Path to the trained YOLO model weights",
    )

    args = parser.parse_args()

    try:
        pipeline = InferencePipeline(args.model)
        (
            all_identifications,
            high_confidence_results,
            yolo_detections,
        ) = await pipeline.run(args.image_path)

        # --- Result Processing and Saving ---
        if high_confidence_results:
            image = cv2.imread(args.image_path)
            image_name = Path(args.image_path).stem
            output_dir = Path("runs/predict") / image_name

            save_results(
                output_dir,
                image,
                all_identifications,
                high_confidence_results,
                yolo_detections,
            )

            print("\n=== Detection Results (Console) ===")
            for i, result in enumerate(high_confidence_results):
                pred = result["brickognize_prediction"]
                print(
                    f"Detection {i + 1}: {pred['name']} (Confidence: {pred['confidence']:.2f})"
                )
        else:
            print("No high-confidence detections to save.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
