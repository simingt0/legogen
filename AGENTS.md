# LegoGen - Agent Coordination Guide

## Quick Start for Agents

Each module has a `plan.md` with full specifications. Read the plan.md for your assigned module before implementing.

## Module Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE FLOW                                  │
│                                                                             │
│   Web App ──POST /build──▶ Server                                          │
│                              │                                              │
│                              ├──[parallel]──▶ Classifier (image → bricks)  │
│                              │                                              │
│                              └──[parallel]──▶ Meshy (description → OBJ)    │
│                                                │                            │
│                                                ▼                            │
│                                            Voxelizer (OBJ → voxel grid)    │
│                                                │                            │
│                                                ▼                            │
│                                            Builder (voxels + bricks →      │
│                                                     layer instructions)     │
│                                                │                            │
│                              ◀────────────────┘                             │
│                              │                                              │
│   Web App ◀──JSON response──┘                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Assignments

| Module | Location | Status | Interface |
|--------|----------|--------|-----------|
| **Classifier** | `pipeline/classifier/` | STUB (partner implementing) | `classify_bricks(image_bytes) → dict[str, int]` |
| **Meshy** | `pipeline/meshy/` | TODO | `generate_3d_model(description) → str (OBJ path)` |
| **Voxelizer** | `pipeline/voxelizer/` | TODO | `voxelize_mesh(obj_path, size) → np.ndarray` |
| **Builder** | `pipeline/builder/` | TODO | `generate_build_instructions(voxels, bricks) → dict` |
| **Server** | `server/` | SCAFFOLDED | FastAPI app, wires everything together |
| **Web** | `web/` | SCAFFOLDED | HTML/CSS/JS frontend |

## Interface Contracts

### Brick Types (shared constant)
```python
VALID_BRICK_TYPES = ["1x1", "1x2", "1x3", "1x4", "1x6", "2x2", "2x3", "2x4", "2x6"]
```

### Classifier Output
```python
{"2x4": 20, "1x1": 50, "2x2": 25, ...}
```

### Voxelizer Output
```python
np.ndarray of shape (width, depth, height), dtype=bool
# voxels[:, :, 0] = bottom layer
# voxels[:, :, z] = layer z
```

### Builder Output
```python
{
    "success": True,
    "layers": [
        [{"type": "2x4", "x": 0, "y": 0, "rotation": 0}, ...],  # layer 0
        [...],  # layer 1
    ],
    "error": None
}
```

### Server Response
```python
{
    "success": True,
    "layers": [...],
    "error": None,
    "metadata": {
        "voxel_dimensions": [16, 12, 8],
        "total_bricks": 45,
        "attempts": 2
    }
}
```

## Dependencies

All dependencies are in `server/requirements.txt`:
```
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6
httpx>=0.25.0
trimesh>=4.0.0
numpy>=1.24.0
```

Install with: `pip install -r server/requirements.txt`

## Running Locally

1. **Set environment variable:**
   ```bash
   export MESHY_API_KEY=your_key_here
   ```

2. **Start the server:**
   ```bash
   cd /path/to/legogen
   PYTHONPATH=. uvicorn server.main:app --reload --port 8000
   ```

3. **Start the web app:**
   ```bash
   cd web
   python -m http.server 8080
   ```

4. **Open browser:** http://localhost:8080

## Testing Individual Modules

Each module can be tested independently:

```python
# Test voxelizer
from pipeline.voxelizer import voxelize_mesh
voxels = voxelize_mesh("/path/to/test.obj", size=16)
print(voxels.shape)

# Test builder
from pipeline.builder import generate_build_instructions
import numpy as np
voxels = np.ones((4, 4, 2), dtype=bool)
bricks = {"2x4": 10, "1x1": 20}
result = generate_build_instructions(voxels, bricks)
print(result)
```

## Important Notes for Implementers

1. **Read your module's plan.md first** — it has complete specifications
2. **Don't modify interfaces** — other modules depend on them
3. **Replace the stub.py** — keep the same function signatures
4. **Test with edge cases** — empty inputs, boundary conditions
5. **The classifier is a mock** — returns fixed values for now
