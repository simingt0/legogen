"""
LegoGen Server - FastAPI application
See plan.md for full specification
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
from pathlib import Path

# Add parent directory to path so we can import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.classifier import classify_bricks
from pipeline.meshy import generate_3d_model
from pipeline.voxelizer import voxelize_mesh
from pipeline.builder import generate_build_instructions

app = FastAPI(title="LegoGen", description="LEGO build instruction generator")

# CORS setup for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BuildMetadata(BaseModel):
    voxel_dimensions: list[int] | None = None
    total_bricks: int | None = None
    attempts: int = 0


class BuildResponse(BaseModel):
    success: bool
    layers: list[list[dict]] | None = None
    error: str | None = None
    metadata: BuildMetadata = BuildMetadata()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/build", response_model=BuildResponse)
async def build(
    image: UploadFile = File(...),
    description: str = Form(...),
    max_attempts: int = Form(default=5),
    voxel_size: int = Form(default=16),
):
    """
    Generate LEGO build instructions from an image of available bricks
    and a description of what to build.
    """
    # Validate inputs
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    if len(description) > 600:
        raise HTTPException(status_code=400, detail="Description too long (max 600 chars)")

    image_bytes = await image.read()

    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Invalid image file")

    metadata = BuildMetadata()

    try:
        # Step 1 & 2: Run classifier and meshy in parallel
        classifier_task = asyncio.create_task(
            asyncio.to_thread(classify_bricks, image_bytes)
        )
        model_task = asyncio.create_task(
            generate_3d_model(description)
        )

        available_bricks, model_path = await asyncio.gather(
            classifier_task,
            model_task,
        )

        # Step 3: Voxelize the 3D model
        voxel_grid = voxelize_mesh(model_path, voxel_size)
        metadata.voxel_dimensions = list(voxel_grid.shape)

        # Step 4: Try to generate build instructions
        for attempt in range(max_attempts):
            metadata.attempts = attempt + 1

            # Make a copy of available_bricks since builder modifies it
            bricks_copy = available_bricks.copy()
            result = generate_build_instructions(voxel_grid, bricks_copy)

            if result["success"]:
                # Count total bricks used
                total = sum(len(layer) for layer in result["layers"])
                metadata.total_bricks = total

                return BuildResponse(
                    success=True,
                    layers=result["layers"],
                    error=None,
                    metadata=metadata,
                )

        # All attempts failed
        return BuildResponse(
            success=False,
            layers=None,
            error=f"Failed to generate build after {max_attempts} attempts",
            metadata=metadata,
        )

    except FileNotFoundError as e:
        return BuildResponse(
            success=False,
            error=f"File not found: {e}",
            metadata=metadata,
        )
    except ValueError as e:
        return BuildResponse(
            success=False,
            error=str(e),
            metadata=metadata,
        )
    except TimeoutError as e:
        return BuildResponse(
            success=False,
            error=f"Timeout: {e}",
            metadata=metadata,
        )
    except Exception as e:
        return BuildResponse(
            success=False,
            error=f"Unexpected error: {e}",
            metadata=metadata,
        )
