"""
Voxelizer module - converts OBJ mesh to voxel grid
See plan.md for full specification
"""

from pathlib import Path

import numpy as np
import trimesh


def voxelize_mesh(obj_path: str, size: int = 16) -> np.ndarray:
    """
    Convert an OBJ mesh file to a voxel grid.

    Args:
        obj_path: Path to the OBJ file on disk
        size: The largest dimension of the output voxel grid (8-32 typical)

    Returns:
        numpy.ndarray of shape (x, y, z) with dtype=bool
        True = filled voxel, False = empty
        The array is oriented so that:
        - Index 0 (x) = left-right
        - Index 1 (y) = front-back
        - Index 2 (z) = bottom-top (layer 0 is the base)

    Raises:
        FileNotFoundError: If obj_path doesn't exist
        ValueError: If the mesh is empty or invalid
    """
    # Validate file exists
    if not Path(obj_path).exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    # Load the mesh
    try:
        mesh = trimesh.load(obj_path)
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {obj_path}: {e}")

    # Handle case where trimesh returns a Scene instead of a Mesh
    if isinstance(mesh, trimesh.Scene):
        # Extract geometry from scene
        if len(mesh.geometry) == 0:
            raise ValueError("Empty mesh: no geometry in scene")
        # Combine all geometries in the scene
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # Validate mesh has geometry
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("Empty mesh: no vertices")

    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError("Empty mesh: no faces")

    # Get mesh bounds
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    mesh_size = bounds[1] - bounds[0]  # [width, depth, height]

    # Calculate pitch (voxel size) based on largest dimension
    max_dimension = max(mesh_size)
    if max_dimension == 0:
        raise ValueError("Mesh has zero size")

    pitch = max_dimension / size

    # Voxelize the mesh
    try:
        voxel_grid = mesh.voxelized(pitch=pitch)
    except Exception as e:
        raise ValueError(f"Failed to voxelize mesh: {e}")

    # Extract the boolean matrix
    # trimesh returns a VoxelGrid object with a .matrix property
    voxel_matrix = voxel_grid.matrix

    # Ensure it's a boolean array
    if voxel_matrix.dtype != bool:
        voxel_matrix = voxel_matrix.astype(bool)

    print(f"Voxelized mesh: {voxel_matrix.shape} voxels")
    print(f"  Original mesh size: {mesh_size}")
    print(f"  Pitch (voxel size): {pitch:.4f}")
    print(f"  Filled voxels: {voxel_matrix.sum()} / {voxel_matrix.size}")

    return voxel_matrix
