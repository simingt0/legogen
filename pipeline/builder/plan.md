# Builder Module

## Overview
The core algorithm that converts a voxel grid into LEGO build instructions using available bricks. Uses a Monte Carlo approach with randomization to explore different valid placements.

## Position in Pipeline
```
Voxelizer (voxel grid) + Classifier (available bricks) → [BUILDER] → Server (layer instructions)
```
This is the final processing step before returning results.

## Interface

### Function Signature
```python
def generate_build_instructions(
    voxel_grid: np.ndarray,
    available_bricks: dict[str, int],
) -> dict:
    """
    Generate LEGO build instructions from a voxel grid.

    Args:
        voxel_grid: 3D boolean numpy array from voxelizer
                    Shape: (width, depth, height)
                    True = filled voxel, False = empty
        available_bricks: Dict mapping brick types to counts
                          e.g., {"2x4": 12, "1x1": 30}
                          NOTE: This dict will be modified (decremented) during building

    Returns:
        {
            "success": bool,
            "layers": [                    # List of layers, bottom to top
                [                          # Each layer is a list of brick placements
                    {
                        "type": "2x4",     # Brick type
                        "x": 0,            # X position (left edge)
                        "y": 0,            # Y position (front edge)
                        "rotation": 0      # 0 = length along X, 90 = length along Y
                    },
                    ...
                ],
                ...
            ],
            "error": str | None           # Error message if success=False
        }
    """
```

### Example Usage
```python
from pipeline.builder import generate_build_instructions
import numpy as np

# Simple 4x4x2 solid block
voxels = np.ones((4, 4, 2), dtype=bool)
bricks = {"2x4": 10, "2x2": 10, "1x1": 20}

result = generate_build_instructions(voxels, bricks)
if result["success"]:
    for i, layer in enumerate(result["layers"]):
        print(f"Layer {i}: {len(layer)} bricks")
```

### Output Format Details

Each brick placement specifies:
- **type**: One of "1x1", "1x2", "1x3", "1x4", "1x6", "2x2", "2x3", "2x4", "2x6"
- **x**: Integer, left edge position (0 to grid_width-1)
- **y**: Integer, front edge position (0 to grid_depth-1)
- **rotation**: 0 or 90 degrees
  - `rotation=0`: Brick's length extends along X axis
  - `rotation=90`: Brick's length extends along Y axis
  - For square bricks (1x1, 2x2), rotation doesn't matter but should be 0

A "2x4" brick at position (x=2, y=3, rotation=0) covers voxels:
- (2,3), (3,3), (4,3), (5,3) — first row
- (2,4), (3,4), (4,4), (5,4) — second row

A "2x4" brick at position (x=2, y=3, rotation=90) covers voxels:
- (2,3), (3,3) — first row
- (2,4), (3,4) — second row
- (2,5), (3,5) — third row
- (2,6), (3,6) — fourth row

## Dependencies
```
numpy>=1.24.0
```

## Algorithm

### High-Level Approach
```
for each layer z from 0 to height-1:
    while layer has unfilled voxels:
        pick a random unfilled voxel
        try to place largest valid brick (weighted by inventory)
        if brick placed:
            mark voxels as filled
            decrement inventory
            record placement
        else:
            fail (shouldn't happen with 1x1 available)

after all layers:
    verify connectivity (no floating bricks)
    if not connected: return failure

return success with layer instructions
```

### Brick Placement Priority
1. **Prefer larger bricks** — reduces total brick count, sturdier builds
2. **Prefer bricks with more inventory** — avoids running out of key sizes
3. **Add randomization** — Monte Carlo variance for retry attempts

Suggested weighting:
```python
score = brick_area * (1 + 0.1 * log(inventory_count + 1))
# Add small random factor for Monte Carlo variance
score *= random.uniform(0.8, 1.2)
```

### Valid Placement Check
A brick can be placed at (x, y, z, rotation) if:
1. All voxels it would cover are within grid bounds
2. All voxels it would cover are True in voxel_grid (need to be filled)
3. All voxels it would cover are not already claimed by another brick
4. We have at least 1 of that brick type in inventory

### Connectivity Check
After placing all bricks, verify the structure is connected:
1. Start BFS/DFS from any brick in layer 0
2. A brick connects to another if they share an edge (adjacent in X or Y on same layer, or vertically adjacent between layers)
3. All bricks must be reachable from the starting brick

### Brick Dimensions Reference
```python
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
# (width, length) - width is always <= length
# rotation=0: width along Y, length along X
# rotation=90: width along X, length along Y
```

## Implementation Notes

1. **Track claimed voxels per layer**: Use a 2D boolean array per layer to track which voxels have been assigned to a brick.

2. **Brick ordering**: Sort bricks by area descending when trying placements.

3. **Random voxel selection**: For Monte Carlo variance, shuffle the order of unfilled voxels before trying to fill them.

4. **Early termination**: If we run out of a required brick size and can't cover remaining voxels, return failure immediately.

5. **Copy the inventory dict**: The caller passes available_bricks — work on a copy or document that it gets modified.

6. **Layer 0 special case**: Bricks in layer 0 don't need support. All other layers need connectivity to layer below.

## Test Cases

### Test 1: Simple 2x4 area
```python
voxels = np.ones((2, 4, 1), dtype=bool)  # Exactly one 2x4 brick
bricks = {"2x4": 1}
result = generate_build_instructions(voxels, bricks)
assert result["success"]
assert len(result["layers"]) == 1
assert len(result["layers"][0]) == 1
```

### Test 2: Not enough bricks
```python
voxels = np.ones((4, 4, 1), dtype=bool)  # 16 voxels
bricks = {"2x4": 1}  # Only covers 8
result = generate_build_instructions(voxels, bricks)
assert not result["success"]
```

### Test 3: Floating section (should fail connectivity)
```python
# Create voxels with a gap in the middle layer
voxels = np.zeros((4, 4, 3), dtype=bool)
voxels[:, :, 0] = True  # Bottom layer
voxels[:, :, 2] = True  # Top layer (floating!)
bricks = {"2x4": 10, "1x1": 20}
result = generate_build_instructions(voxels, bricks)
assert not result["success"]  # Top layer is floating
```

### Test 4: Complex shape
```python
# L-shaped voxel structure
voxels = np.zeros((4, 4, 2), dtype=bool)
voxels[0:2, :, :] = True
voxels[:, 0:2, :] = True
bricks = {"2x4": 5, "2x2": 5, "1x1": 20}
result = generate_build_instructions(voxels, bricks)
assert result["success"]
```

## Error Handling
- Empty voxel grid (no True values): return `{"success": False, "error": "Empty voxel grid"}`
- Insufficient bricks: return `{"success": False, "error": "Insufficient bricks"}`
- Connectivity failure: return `{"success": False, "error": "Structure has floating sections"}`
