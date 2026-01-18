"""
Builder module - converts voxel grid to LEGO build instructions
See plan.md for full specification
"""
import numpy as np

# Brick dimensions: (width, length) where width <= length
BRICK_DIMS = {
    "1x1": (1, 1),
    "1x2": (1, 2),
    "1x3": (1, 3),
    "1x4": (1, 4),
    "1x6": (1, 6),
    "2x2": (2, 2),
    "2x3": (2, 3),
    "2x4": (2, 4),
    "2x6": (2, 6),
}

# Sorted by area descending for placement priority
BRICKS_BY_SIZE = sorted(BRICK_DIMS.keys(), key=lambda b: BRICK_DIMS[b][0] * BRICK_DIMS[b][1], reverse=True)


def generate_build_instructions(
    voxel_grid: np.ndarray,
    available_bricks: dict[str, int],
) -> dict:
    """
    Generate LEGO build instructions from a voxel grid.

    Args:
        voxel_grid: 3D boolean numpy array (width, depth, height)
        available_bricks: Dict mapping brick types to counts

    Returns:
        {
            "success": bool,
            "layers": [...] or None,
            "error": str or None
        }
    """
    raise NotImplementedError("See plan.md for implementation details")
