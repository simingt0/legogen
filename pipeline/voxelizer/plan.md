# Voxelizer Module

## Overview
Converts a 3D mesh (OBJ format) into a voxel grid (3D boolean array). This is a pure Python module using the `trimesh` library — no background processes or external apps required.

## Position in Pipeline
```
Meshy (OBJ file) → [VOXELIZER] → Builder (voxel grid)
```

## Interface

### Function Signature
```python
def voxelize_mesh(obj_path: str, size: int = 16) -> np.ndarray:
    """
    Convert an OBJ mesh file to a voxel grid.

    Args:
        obj_path: Path to the OBJ file on disk
        size: The largest dimension of the output voxel grid (8-32 typical)
              The mesh will be scaled to fit within a cube of this size.

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
```

### Example Usage
```python
from pipeline.voxelizer import voxelize_mesh
import numpy as np

voxels = voxelize_mesh("/tmp/model.obj", size=16)
# voxels.shape might be (16, 12, 10) for a non-cubic model
# voxels[0, 0, 0] is the bottom-left-front corner
# voxels[:, :, 0] is the entire bottom layer
```

### Output Format Details
The voxel grid is a 3D numpy boolean array where:
- Dimensions are (width, depth, height)
- `voxels[:, :, z]` gives you layer `z` as a 2D slice
- The mesh is centered in the grid
- At least one dimension equals `size`, others may be smaller

## Dependencies
```
trimesh>=4.0.0
numpy>=1.24.0
```

## Implementation Notes

1. **Load mesh**: Use `trimesh.load(obj_path)`

2. **Voxelize**: Use `mesh.voxelized(pitch)` where pitch = mesh_size / desired_voxel_count
   - Calculate pitch so the largest dimension becomes `size` voxels

3. **Extract matrix**: The `VoxelGrid` object has a `.matrix` property that gives the boolean array

4. **Orientation**: Ensure Z is up (trimesh default). If the model appears sideways, you may need to rotate.

5. **Hollow vs Filled**: `mesh.voxelized()` gives a filled voxelization. For LEGO purposes, we want filled (solid bricks, not shells).

## Test Cases

### Test 1: Simple cube
```python
# Create a unit cube mesh, voxelize at size=4
# Expected: 4x4x4 array, all True
```

### Test 2: Rectangular prism
```python
# Create a 2:1:1 rectangular mesh, voxelize at size=8
# Expected: 8x4x4 array (longest dimension = 8)
```

### Test 3: Sphere
```python
# Create a sphere, voxelize at size=10
# Expected: ~10x10x10 array with spherical pattern of True values
```

## Error Handling
- If file doesn't exist: raise `FileNotFoundError`
- If mesh has no geometry: raise `ValueError("Empty mesh")`
- If mesh can't be voxelized: raise `ValueError` with details
