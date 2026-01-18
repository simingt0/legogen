"""
Voxelizer module - converts OBJ mesh to voxel grid
See plan.md for full specification
"""

from collections import deque
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

    # Calculate pitch based on size-1 to ensure we don't exceed requested size
    pitch = max_dimension / (size - 1)

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

    # Swap Y and Z axes so Z is vertical (up)
    # trimesh voxelizes with Y as up, but we need Z as up for layer-by-layer building
    voxel_matrix = np.transpose(voxel_matrix, (0, 2, 1))

    # Extract only the largest connected component
    voxel_matrix = _extract_largest_component(voxel_matrix)

    # Fill enclosed spaces (but keep holes/openings)
    voxel_matrix = _fill_enclosed_spaces(voxel_matrix)

    return voxel_matrix


def _extract_largest_component(voxels: np.ndarray) -> np.ndarray:
    """
    Extract only the largest connected component from the voxel grid.
    Voxels are considered connected if they share a face (6-connectivity).

    Returns:
        New voxel grid containing only the largest connected component
    """
    if not np.any(voxels):
        return voxels

    visited = np.zeros_like(voxels, dtype=bool)
    components = []

    # Find all connected components
    for x in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for z in range(voxels.shape[2]):
                if voxels[x, y, z] and not visited[x, y, z]:
                    # BFS to find this component
                    component = _bfs_component(voxels, visited, (x, y, z))
                    components.append(component)

    if not components:
        return voxels

    # Find largest component
    largest = max(components, key=len)

    # Create new voxel grid with only largest component
    result = np.zeros_like(voxels, dtype=bool)
    for x, y, z in largest:
        result[x, y, z] = True

    return result


def _bfs_component(voxels: np.ndarray, visited: np.ndarray, start: tuple) -> list:
    """
    BFS to find all voxels in a connected component starting from `start`.

    Returns:
        List of (x, y, z) tuples in the component
    """
    component = []
    queue = deque([start])
    visited[start] = True

    width, depth, height = voxels.shape

    while queue:
        x, y, z = queue.popleft()
        component.append((x, y, z))

        # Check 6 neighbors (face-connected)
        for dx, dy, dz in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            nx, ny, nz = x + dx, y + dy, z + dz

            # Check bounds
            if 0 <= nx < width and 0 <= ny < depth and 0 <= nz < height:
                # Check if filled and not visited
                if voxels[nx, ny, nz] and not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))

    return component


def _fill_enclosed_spaces(voxels: np.ndarray) -> np.ndarray:
    """
    Fill completely enclosed spaces in the voxel grid.
    Uses flood fill from outside - anything not reachable from outside is enclosed.

    Returns:
        Voxel grid with enclosed spaces filled
    """
    if not np.any(voxels):
        return voxels

    width, depth, height = voxels.shape

    # Create a grid to track what's reachable from outside
    reachable = np.zeros_like(voxels, dtype=bool)

    # Start flood fill from all border voxels that are empty
    # These represent the "outside" of the model
    queue = deque()

    # Add all empty border cells to queue
    for x in range(width):
        for y in range(depth):
            for z in range(height):
                # Check if on border
                on_border = (
                    x == 0
                    or x == width - 1
                    or y == 0
                    or y == depth - 1
                    or z == 0
                    or z == height - 1
                )

                if on_border and not voxels[x, y, z]:
                    queue.append((x, y, z))
                    reachable[x, y, z] = True

    # Flood fill to mark all reachable empty spaces
    while queue:
        x, y, z = queue.popleft()

        # Check all 6 neighbors
        for dx, dy, dz in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            nx, ny, nz = x + dx, y + dy, z + dz

            # Check bounds
            if 0 <= nx < width and 0 <= ny < depth and 0 <= nz < height:
                # If empty and not yet marked as reachable
                if not voxels[nx, ny, nz] and not reachable[nx, ny, nz]:
                    reachable[nx, ny, nz] = True
                    queue.append((nx, ny, nz))

    # Create result: filled voxels + unreachable empty spaces (enclosed)
    result = voxels.copy()
    for x in range(width):
        for y in range(depth):
            for z in range(height):
                # If empty and not reachable from outside, it's enclosed - fill it
                if not voxels[x, y, z] and not reachable[x, y, z]:
                    result[x, y, z] = True

    return result
