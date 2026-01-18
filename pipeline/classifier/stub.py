"""
Classifier module - identifies LEGO bricks from an image
Uses the inference pipeline script from model/inference_pipeline.py
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

VALID_BRICK_TYPES = [
    "1x1",
    "1x2",
    "1x3",
    "1x4",
    "1x6",
    "2x2",
    "2x3",
    "2x4",
    "2x6",
]

# Mapping from Brickognize brick names/part numbers to our standard format
BRICK_NAME_MAPPING = {
    # 1xN bricks
    "3005": "1x1",  # Brick 1x1
    "3004": "1x2",  # Brick 1x2
    "3622": "1x3",  # Brick 1x3
    "3010": "1x4",  # Brick 1x4
    "3009": "1x6",  # Brick 1x6
    # 2xN bricks
    "3003": "2x2",  # Brick 2x2
    "3002": "2x3",  # Brick 2x3
    "3001": "2x4",  # Brick 2x4
    "2456": "2x6",  # Brick 2x6
    "44237": "2x6",  # Brick 2x6 (alternate)
    # Plates (treat as bricks)
    "3024": "1x1",  # Plate 1x1
    "3023": "1x2",  # Plate 1x2
    "3623": "1x3",  # Plate 1x3
    "3710": "1x4",  # Plate 1x4
    "3666": "1x6",  # Plate 1x6
    "3022": "2x2",  # Plate 2x2
    "3021": "2x3",  # Plate 2x3
    "3020": "2x4",  # Plate 2x4
    "3958": "2x6",  # Plate 2x6
}


def _parse_brick_name_to_type(brick_name: str) -> str | None:
    """
    Parse a brick name from Brickognize and convert to our standard format.

    Args:
        brick_name: Brick name like "Brick 2x4" or part number like "3001"

    Returns:
        Standard brick type like "2x4" or None if not recognized
    """
    if not brick_name:
        return None

    # Check if it's a part number we know
    for part_num, brick_type in BRICK_NAME_MAPPING.items():
        if part_num in str(brick_name):
            return brick_type

    # Try to extract dimensions from name like "Brick 2x4" or "Plate 1x2"
    import re

    match = re.search(r"(\d+)\s*x\s*(\d+)", brick_name.lower())
    if match:
        w, l = match.groups()
        brick_type = f"{w}x{l}"
        if brick_type in VALID_BRICK_TYPES:
            return brick_type

    return None


def classify_bricks(image_bytes: bytes) -> dict[str, int]:
    """
    Analyze an image and return counts of each brick type detected.

    Uses inference_pipeline.py script with YOLO + Brickognize.
    Caches results based on image hash to avoid reprocessing.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG or PNG)

    Returns:
        Dictionary mapping brick type strings to counts.
    """
    # Validate that we received some data
    if not image_bytes or len(image_bytes) < 100:
        raise ValueError("Cannot decode image: insufficient data")

    # Check cache first
    image_hash = hashlib.md5(image_bytes).hexdigest()
    cached_result = _get_cached_classification(image_hash)
    if cached_result is not None:
        print(f"   Using cached classification for image (hash: {image_hash[:8]}...)")
        return cached_result

    # Use the real inference pipeline - no fallback
    result = _run_inference_pipeline(image_bytes)

    # Cache the result
    _cache_classification(image_hash, result)

    return result


def _run_inference_pipeline(image_bytes: bytes) -> dict[str, int]:
    """
    Run the inference pipeline script on image bytes.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Brick inventory dict
    """
    # Find the inference pipeline script
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "model" / "inference_pipeline.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Inference pipeline not found at {script_path}")

    # Find the YOLO model
    model_path = _find_yolo_model()
    if model_path is None:
        raise FileNotFoundError(
            "YOLO model not found. Please ensure the model exists at model/weights/best.pt "
            "or set YOLO_MODEL_PATH environment variable."
        )

    # Save image bytes to temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_image:
        tmp_image.write(image_bytes)
        tmp_image_path = tmp_image.name

    try:
        # Run the inference pipeline script (silently)
        cmd = [
            sys.executable,
            str(script_path),
            tmp_image_path,
            "--model",
            model_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(project_root / "model"),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Inference pipeline failed: {result.stderr}")

        # Parse the output JSON file
        image_name = Path(tmp_image_path).stem
        output_dir = project_root / "model" / "runs" / "predict" / image_name
        json_path = output_dir / "results.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Results JSON not found at {json_path}")

        with open(json_path, "r") as f:
            detections = json.load(f)

        # Convert to inventory
        inventory = _convert_detections_to_inventory(detections)

        total_bricks = sum(inventory.values())
        if total_bricks == 0:
            raise ValueError(
                "No bricks detected in image. Please ensure the image contains visible LEGO bricks."
            )

        return inventory

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_image_path)
        except:
            pass


def _find_yolo_model() -> str | None:
    """
    Find the YOLO model file.

    Returns:
        Path to the model file, or None if not found
    """
    project_root = Path(__file__).parent.parent.parent

    # Check common locations
    possible_paths = [
        # Environment variable
        os.environ.get("YOLO_MODEL_PATH"),
        # In model/runs directory
        project_root / "model" / "runs" / "detect" / "train" / "weights" / "best.pt",
        project_root
        / "model"
        / "runs"
        / "detect"
        / "models"
        / "yolov8_s_20260118_004503"
        / "train"
        / "weights"
        / "best.pt",
        # Direct in model directory
        project_root / "model" / "best.pt",
        project_root / "model" / "weights" / "best.pt",
    ]

    for path in possible_paths:
        if path and Path(path).exists():
            return str(path)

    return None


def _convert_detections_to_inventory(detections: list) -> dict[str, int]:
    """
    Convert Brickognize detections to our brick inventory format.

    Args:
        detections: List of detection dicts from inference pipeline JSON

    Returns:
        Inventory dict mapping brick types to counts
    """
    inventory = {brick_type: 0 for brick_type in VALID_BRICK_TYPES}

    for detection in detections:
        pred = detection.get("brickognize_prediction", {})
        brick_name = pred.get("name", "")
        part_number = pred.get("part_number", "")

        # Try to parse the brick type
        brick_type = _parse_brick_name_to_type(brick_name)
        if brick_type is None and part_number:
            brick_type = _parse_brick_name_to_type(part_number)

        if brick_type and brick_type in VALID_BRICK_TYPES:
            inventory[brick_type] += 1

    return inventory


def _get_cache_dir() -> Path:
    """Get the cache directory for classifier results."""
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _get_cached_classification(image_hash: str) -> dict[str, int] | None:
    """
    Get cached classification results for an image hash.

    Args:
        image_hash: MD5 hash of the image bytes

    Returns:
        Cached brick inventory or None if not cached
    """
    cache_file = _get_cache_dir() / f"{image_hash}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _cache_classification(image_hash: str, inventory: dict[str, int]) -> None:
    """
    Cache classification results for an image hash.

    Args:
        image_hash: MD5 hash of the image bytes
        inventory: Brick inventory to cache
    """
    cache_file = _get_cache_dir() / f"{image_hash}.json"

    try:
        with open(cache_file, "w") as f:
            json.dump(inventory, f, indent=2)
    except IOError as e:
        print(f"Warning: Failed to cache classification: {e}")
