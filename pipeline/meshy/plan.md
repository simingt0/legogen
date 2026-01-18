# Meshy Module

## Overview
Generates a 3D model from a text description using the Meshy API. This is an async module that creates a preview task, polls for completion, and downloads the resulting OBJ file.

## Position in Pipeline
```
Server (description) → [MESHY] → Voxelizer (OBJ file)
```
Runs in **parallel** with the Classifier.

## Interface

### Function Signature
```python
async def generate_3d_model(
    description: str,
    output_dir: str = "/tmp/legogen",
    art_style: str = "sculpture",
    timeout: int = 300,
) -> str:
    """
    Generate a 3D model from a text description using Meshy API.

    Args:
        description: Text description of what to create (max 600 chars)
        output_dir: Directory to save the downloaded OBJ file
        art_style: "sculpture" (blocky, better for voxelization) or "realistic"
        timeout: Max seconds to wait for generation (default 5 minutes)

    Returns:
        Path to the downloaded OBJ file (e.g., "/tmp/legogen/abc123.obj")

    Raises:
        ValueError: If description is empty or too long
        TimeoutError: If generation takes longer than timeout
        RuntimeError: If API request fails or returns error
    """
```

### Example Usage
```python
from pipeline.meshy import generate_3d_model
import asyncio

async def main():
    obj_path = await generate_3d_model("a small house with a chimney")
    print(f"Model saved to: {obj_path}")

asyncio.run(main())
```

### Configuration
The API key should be read from environment variable:
```python
MESHY_API_KEY = os.environ.get("MESHY_API_KEY")
```

## Dependencies
```
httpx>=0.25.0  # async HTTP client
```

## Meshy API Details

### Base URL
```
https://api.meshy.ai
```

### Authentication
```
Authorization: Bearer {MESHY_API_KEY}
```

### Step 1: Create Preview Task
```
POST /openapi/v2/text-to-3d
Content-Type: application/json

{
    "mode": "preview",
    "prompt": "a small house with a chimney",
    "art_style": "sculpture",
    "ai_model": "meshy-5"
}

Response:
{
    "result": "task_id_string"
}
```

### Step 2: Poll for Completion
```
GET /openapi/v2/text-to-3d/{task_id}

Response:
{
    "id": "task_id_string",
    "status": "PENDING" | "IN_PROGRESS" | "SUCCEEDED" | "FAILED",
    "progress": 0-100,
    "model_urls": {
        "obj": "https://...",
        "glb": "https://...",
        ...
    }
}
```

Poll every 5 seconds until status is "SUCCEEDED" or "FAILED".

### Step 3: Download OBJ
Once status is "SUCCEEDED", download the file from `model_urls.obj`.

## Implementation Notes

1. **Use httpx for async HTTP**: Cleaner than aiohttp for this use case.

2. **Polling strategy**:
   - Start with 5-second intervals
   - Log progress for debugging
   - Respect timeout parameter

3. **Art style**: Use "sculpture" by default — produces blockier models that voxelize better than "realistic".

4. **File naming**: Use the task ID as filename: `{task_id}.obj`

5. **Directory creation**: Create output_dir if it doesn't exist.

6. **Error handling**:
   - Check for API errors in responses
   - Handle network failures gracefully
   - Clean up partial downloads on failure

## Mock Implementation (for testing without API)
```python
async def generate_3d_model(description: str, ...) -> str:
    """Mock that returns a path to a test OBJ file."""
    # Could return path to a simple cube OBJ for testing
    return "/path/to/test/cube.obj"
```

## Test Cases

### Test 1: Simple object
```python
# Input: "a cube"
# Expected: Returns path to valid OBJ file
```

### Test 2: Timeout
```python
# Input: description with timeout=1 (too short)
# Expected: TimeoutError
```

### Test 3: Invalid API key
```python
# Input: Valid description, bad API key
# Expected: RuntimeError with auth error message
```

### Test 4: Empty description
```python
# Input: ""
# Expected: ValueError
```

## Error Handling
- Empty description: raise `ValueError("Description cannot be empty")`
- Description > 600 chars: raise `ValueError("Description too long (max 600 chars)")`
- API auth failure: raise `RuntimeError("Meshy API authentication failed")`
- API error response: raise `RuntimeError(f"Meshy API error: {details}")`
- Timeout: raise `TimeoutError(f"Model generation timed out after {timeout}s")`
- Download failure: raise `RuntimeError("Failed to download model file")`
