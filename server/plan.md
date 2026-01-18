# Server Module

## Overview
FastAPI server that exposes the LegoGen API. Handles HTTP requests, orchestrates the pipeline, and returns JSON responses to the web frontend.

## Architecture
```
Web App (localhost:8080)
    │
    ▼
┌─────────────────────────────────────────┐
│  FastAPI Server (localhost:8000)        │
│  ├── POST /build                        │
│  │   ├── [parallel] classify_bricks()   │
│  │   └── [parallel] generate_3d_model() │
│  │         └── voxelize_mesh()          │
│  │               └── generate_build()   │
│  │                     └── (retry x5)   │
│  └── GET /health                        │
└─────────────────────────────────────────┘
```

## Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

### POST /build
Main endpoint to generate LEGO build instructions.

**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `image`: File upload (JPEG/PNG of LEGO bricks)
  - `description`: String (what to build, max 600 chars)

**Response (success):**
```json
{
    "success": true,
    "layers": [
        [
            {"type": "2x4", "x": 0, "y": 0, "rotation": 0},
            {"type": "2x2", "x": 4, "y": 0, "rotation": 0}
        ],
        [
            {"type": "2x4", "x": 1, "y": 0, "rotation": 0}
        ]
    ],
    "error": null,
    "metadata": {
        "voxel_dimensions": [8, 6, 4],
        "total_bricks": 15,
        "attempts": 2
    }
}
```

**Response (failure):**
```json
{
    "success": false,
    "layers": null,
    "error": "Failed to generate build after 5 attempts",
    "metadata": {
        "voxel_dimensions": [8, 6, 4],
        "total_bricks": null,
        "attempts": 5
    }
}
```

## Interface

### Main Application
```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LegoGen")

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Pipeline Orchestration
```python
async def run_pipeline(
    image_bytes: bytes,
    description: str,
    max_attempts: int = 5,
    voxel_size: int = 16,
) -> dict:
    """
    Orchestrate the full pipeline.

    1. Run classifier and meshy in parallel
    2. Voxelize the 3D model
    3. Try to generate build instructions (up to max_attempts)
    4. Return result
    """
```

## Dependencies
```
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6  # Required for file uploads
```

## Implementation Notes

1. **Parallel execution**: Use `asyncio.gather()` to run classifier and meshy concurrently:
   ```python
   classifier_task = asyncio.create_task(asyncio.to_thread(classify_bricks, image_bytes))
   model_task = asyncio.create_task(generate_3d_model(description))
   available_bricks, model_path = await asyncio.gather(classifier_task, model_task)
   ```

2. **Voxel size selection**: Could be based on brick count. More bricks = larger model. Start with fixed 16.

3. **Retry loop**: The builder has randomization, so retrying may succeed even if first attempt fails.

4. **Error handling**: Catch exceptions from pipeline modules and return appropriate error responses.

5. **Logging**: Add logging for debugging during hackathon.

## File Structure
```
server/
├── plan.md
├── main.py          # FastAPI app + endpoints
└── requirements.txt
```

## Running the Server
```bash
cd server
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Or from project root:
```bash
PYTHONPATH=. uvicorn server.main:app --reload --port 8000
```

## Environment Variables
```
MESHY_API_KEY=your_api_key_here
```

## Test Cases

### Test 1: Health check
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

### Test 2: Valid build request
```bash
curl -X POST http://localhost:8000/build \
  -F "image=@bricks.jpg" \
  -F "description=a small cube"
# Expected: JSON with success=true and layers
```

### Test 3: Missing image
```bash
curl -X POST http://localhost:8000/build \
  -F "description=a cube"
# Expected: 422 Unprocessable Entity
```

### Test 4: Empty description
```bash
curl -X POST http://localhost:8000/build \
  -F "image=@bricks.jpg" \
  -F "description="
# Expected: Error response (empty description)
```

## Error Handling
- Missing required fields: FastAPI returns 422 automatically
- Pipeline errors: Catch and return `{"success": false, "error": "..."}`
- Meshy API failures: Return descriptive error
- Timeout: Return error after reasonable wait
