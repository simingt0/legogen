"""
Voxelizer module - converts OBJ mesh to voxel grid
See plan.md for full specification
"""
import numpy as np


def voxelize_mesh(obj_path: str, size: int = 16) -> np.ndarray:
    """
    Convert an OBJ mesh file to a voxel grid.

    Args:
        obj_path: Path to the OBJ file on disk
        size: The largest dimension of the output voxel grid (8-32 typical)

    Returns:
        numpy.ndarray of shape (x, y, z) with dtype=bool
        True = filled voxel, False = empty
    """
    raise NotImplementedError("See plan.md for implementation details")
