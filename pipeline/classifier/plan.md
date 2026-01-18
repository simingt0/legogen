# Classifier Module

## Overview
Takes an image of LEGO bricks and identifies the types and quantities of bricks available. This module will be implemented by a partner — this spec defines the interface contract.

## Position in Pipeline
```
Web App (image upload) → Server → [CLASSIFIER] → Builder (available bricks)
```
Runs in **parallel** with the Meshy → Voxelizer chain.

## Interface

### Function Signature
```python
def classify_bricks(image_bytes: bytes) -> dict[str, int]:
    """
    Analyze an image and return counts of each brick type detected.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG or PNG)

    Returns:
        Dictionary mapping brick type strings to counts.
        Brick types use the format "{width}x{length}" where width <= length.

        Example: {"2x4": 12, "1x1": 30, "2x2": 8, "1x4": 5}

    Raises:
        ValueError: If image cannot be decoded or processed
    """
```

### Valid Brick Types
The classifier should only return these brick types (standard LEGO bricks):
```python
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
```

All bricks are standard height (not plates, not slopes). Width is always listed first, and width ≤ length.

### Example Usage
```python
from pipeline.classifier import classify_bricks

with open("bricks.jpg", "rb") as f:
    image_bytes = f.read()

available = classify_bricks(image_bytes)
# Returns: {"2x4": 12, "1x1": 30, "2x2": 8, ...}
```

### Output Format Details
- Keys are strings in "{width}x{length}" format
- Values are positive integers (count of that brick type)
- Missing brick types should be omitted (not included with 0)
- If no bricks detected, return empty dict `{}`

## Dependencies
```
# Partner to specify based on their implementation
# Likely: opencv-python, torch, ultralytics, or similar
```

## Implementation Notes (for partner)

1. **Input handling**: Image bytes could be JPEG or PNG. Use PIL or OpenCV to decode.

2. **Detection approach options**:
   - YOLO/object detection model trained on LEGO bricks
   - Classification + counting
   - Color segmentation + shape matching

3. **Brick orientation**: Bricks may be photographed from various angles. The classifier should handle top-down and angled views.

4. **Color**: Colors don't matter for MVP — just identify shape/size.

## Mock Implementation (for testing other modules)
```python
def classify_bricks(image_bytes: bytes) -> dict[str, int]:
    """Mock implementation that returns a fixed set of bricks."""
    return {
        "2x4": 20,
        "2x3": 15,
        "2x2": 25,
        "1x4": 20,
        "1x3": 15,
        "1x2": 30,
        "1x1": 50,
    }
```

## Test Cases

### Test 1: Valid image
```python
# Input: JPEG bytes of bricks on a table
# Expected: dict with brick counts
```

### Test 2: Empty/no bricks
```python
# Input: Image with no LEGO bricks
# Expected: {} (empty dict)
```

### Test 3: Invalid image
```python
# Input: Random bytes, not a valid image
# Expected: ValueError
```

## Error Handling
- Invalid image data: raise `ValueError("Cannot decode image")`
- Processing failure: raise `ValueError` with description
