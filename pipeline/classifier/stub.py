"""
Classifier module - identifies LEGO bricks from an image
See plan.md for full specification

This is a MOCK implementation for testing. Partner will replace with real classifier.
"""

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


def classify_bricks(image_bytes: bytes) -> dict[str, int]:
    """
    Analyze an image and return counts of each brick type detected.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG or PNG)

    Returns:
        Dictionary mapping brick type strings to counts.

    NOTE: This is a MOCK implementation that returns fixed values.
    """
    # Validate that we received some data
    if not image_bytes or len(image_bytes) < 100:
        raise ValueError("Cannot decode image: insufficient data")

    # Mock response with a reasonable set of bricks
    return {
        "2x6": 10,
        "2x4": 20,
        "2x3": 15,
        "2x2": 25,
        "1x6": 10,
        "1x4": 20,
        "1x3": 15,
        "1x2": 30,
        "1x1": 50,
    }
