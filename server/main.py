"""
LegoGen Server - FastAPI application
Integrates all pipeline components: classifier, meshy, voxelizer, and builder
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path so we can import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.builder import BRICK_DIMS, generate_build_instructions
from pipeline.classifier import classify_bricks
from pipeline.meshy import generate_3d_model
from pipeline.voxelizer import voxelize_mesh

# Color codes for terminal output
BRICK_COLORS = [
    "\033[91m",  # Bright Red
    "\033[92m",  # Bright Green
    "\033[93m",  # Bright Yellow
    "\033[94m",  # Bright Blue
    "\033[95m",  # Bright Magenta
    "\033[96m",  # Bright Cyan
    "\033[97m",  # Bright White
    "\033[31m",  # Red
    "\033[32m",  # Green
    "\033[33m",  # Yellow
    "\033[34m",  # Blue
    "\033[35m",  # Magenta
    "\033[36m",  # Cyan
]

BRICK_PATTERNS = ["█", "▓", "▒", "░", "▪", "▫", "●", "○", "◆", "◇", "■", "□"]

RESET_COLOR = "\033[0m"

app = FastAPI(title="LegoGen", description="LEGO build instruction generator")

# CORS setup for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
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
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/build", response_model=BuildResponse)
async def build(request: Request):
    """
    Generate LEGO build instructions from an image of available bricks
    and a description of what to build.

    Pipeline:
    1. Classify bricks from image (parallel)
    2. Generate 3D model from description (parallel)
    3. Voxelize the 3D model
    4. Generate build instructions using available bricks

    Args:
        image: Image file containing LEGO bricks
        description: Text description of what to build (max 600 chars)
        max_attempts: Maximum attempts for builder algorithm (default 5)
        voxel_size: Size of voxel grid (default 16)

    Returns:
        BuildResponse with layers and metadata
    """
    # Parse multipart form data manually
    try:
        form = await request.form()
        description = form.get("description")
        image = form.get("image")
        voxel_size = int(form.get("voxel_size", 10))
        max_attempts = int(form.get("max_attempts", 5))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse form data: {e}")

    # Validate inputs
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    if len(description) > 600:
        raise HTTPException(
            status_code=400, detail="Description too long (max 600 chars)"
        )

    if not image:
        raise HTTPException(status_code=400, detail="Image file required")

    # Read image bytes
    image_bytes = await image.read()

    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Invalid image file")

    metadata = BuildMetadata()

    # Track start time for minimum delay (let users enjoy the mini-game!)
    start_time = time.time()
    MIN_RESPONSE_TIME = 5.0  # seconds

    try:
        print(f"\n{'=' * 70}")
        print(f"🚀 STARTING PIPELINE")
        print(f"{'=' * 70}")
        print(f"Description: '{description}'")
        print(f"Image size: {len(image_bytes):,} bytes")
        print(f"Voxel size: {voxel_size}, Max attempts: {max_attempts}")
        print(f"{'=' * 70}\n")

        # Step 1 & 2: Run classifier and meshy in parallel
        print("📋 Step 1 & 2: Running classifier and Meshy in parallel...")

        classifier_task = asyncio.create_task(
            asyncio.to_thread(classify_bricks, image_bytes)
        )

        # Use test mode by default (will use cache if available)
        meshy_mode = os.environ.get("MESHY_MODE", "test")
        model_task = asyncio.create_task(
            generate_3d_model(
                description=description, output_dir="/tmp/legogen", mode=meshy_mode
            )
        )

        available_bricks, model_path = await asyncio.gather(
            classifier_task,
            model_task,
        )

        total_bricks = sum(available_bricks.values())
        print(f"   ✅ Classifier: {total_bricks} bricks detected")
        print(f"   ✅ Meshy: 3D model ready\n")

        # Step 3: Voxelize the 3D model
        print(f"🔲 Step 3: Voxelizing model (size={voxel_size})...")
        voxel_grid = voxelize_mesh(model_path, voxel_size)
        metadata.voxel_dimensions = list(voxel_grid.shape)
        filled_voxels = voxel_grid.sum()
        print(f"   ✅ Voxelized: {voxel_grid.shape}, {filled_voxels:,} voxels filled\n")

        # Step 4: Generate build instructions (Algorithm 7 handles retries internally)
        print(f"🧱 Step 4: Generating build instructions...")

        result = generate_build_instructions(voxel_grid, available_bricks.copy())
        metadata.attempts = 1

        if result["success"]:
            # Count total bricks used
            total = sum(len(layer) for layer in result["layers"])
            metadata.total_bricks = total

            print(f"   ✅ Success!")

            # Print color-coded visualization
            _print_build_visualization(result["layers"], voxel_grid.shape)

            print(f"\n{'=' * 70}")
            print(f"✅ BUILD COMPLETE")
            print(f"{'=' * 70}")
            print(f"Layers: {len(result['layers'])} | Bricks: {total}")
            if "total_cells_skipped" in result:
                print(
                    f"Total cells skipped: {result['total_cells_skipped']} (tolerance: {result.get('total_tolerance', 0)})"
                )
            print(f"{'=' * 70}\n")

            # Ensure minimum response time so users can enjoy the mini-game
            elapsed = time.time() - start_time
            if elapsed < MIN_RESPONSE_TIME:
                delay = MIN_RESPONSE_TIME - elapsed
                print(f"⏳ Adding {delay:.1f}s delay for mini-game enjoyment...")
                await asyncio.sleep(delay)

            return BuildResponse(
                success=True,
                layers=result["layers"],
                error=None,
                metadata=metadata,
            )

        # Build failed
        error_msg = result.get("error", "Failed to generate build")

        print(f"\n{'=' * 70}")
        print(f"❌ BUILD FAILED")
        print(f"{'=' * 70}")
        print(f"Reason: Not enough bricks or model too complex")
        print(f"{'=' * 70}\n")

        return BuildResponse(
            success=False,
            layers=None,
            error=error_msg,
            metadata=metadata,
        )

    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        print(f"\n❌ ERROR: {error_msg}\n")
        return BuildResponse(
            success=False,
            error=error_msg,
            metadata=metadata,
        )
    except ValueError as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}\n")
        return BuildResponse(
            success=False,
            error=error_msg,
            metadata=metadata,
        )
    except TimeoutError as e:
        error_msg = f"Timeout: {e}"
        print(f"\n❌ ERROR: {error_msg}\n")
        return BuildResponse(
            success=False,
            error=error_msg,
            metadata=metadata,
        )
    except RuntimeError as e:
        # Meshy API errors
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}\n")
        return BuildResponse(
            success=False,
            error=error_msg,
            metadata=metadata,
        )
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"\n❌ ERROR: {error_msg}")
        import traceback

        traceback.print_exc()
        print()
        return BuildResponse(
            success=False,
            error=error_msg,
            metadata=metadata,
        )


def _print_build_visualization(layers: list, voxel_shape: tuple):
    """Print color-coded visualization of build instructions."""
    width, depth, height = voxel_shape

    print(f"\n{'=' * 70}")
    print("BUILD INSTRUCTIONS VISUALIZATION")
    print(f"{'=' * 70}\n")

    for z, layer in enumerate(layers):
        if len(layer) == 0:
            continue

        print(f">>> LAYER {z} <<<")
        print(f"    Bricks: {len(layer)}")

        # Create visual grid
        display = [[-1 for _ in range(width)] for _ in range(depth)]
        color_map = {}

        # Assign colors to each brick
        for i, brick in enumerate(layer):
            color = BRICK_COLORS[i % len(BRICK_COLORS)]
            pattern = BRICK_PATTERNS[i % len(BRICK_PATTERNS)]
            color_map[i] = (color, pattern)

            brick_w, brick_l = BRICK_DIMS[brick["type"]]

            if brick["rotation"] == 0:
                dx, dy = brick_l, brick_w
            else:
                dx, dy = brick_w, brick_l

            # Mark cells with brick index
            for j in range(dx):
                for k in range(dy):
                    x_pos = brick["x"] + j
                    y_pos = brick["y"] + k
                    if 0 <= x_pos < width and 0 <= y_pos < depth:
                        display[y_pos][x_pos] = i

        # Print visual grid
        print("\n    " + "─" * (width * 2 + 1))
        for y in range(depth):
            row_str = "    │"
            for x in range(width):
                cell = display[y][x]
                if cell == -1:
                    row_str += " │"
                else:
                    color, pattern = color_map[cell]
                    row_str += f"{color}{pattern}{RESET_COLOR}│"
            print(row_str)
            print("    " + "─" * (width * 2 + 1))

        # Print legend
        print("\n    Legend:")
        for i, brick in enumerate(layer):
            color, pattern = color_map[i]
            brick_type = brick["type"]
            x, y = brick["x"], brick["y"]
            rotation = brick["rotation"]

            brick_w, brick_l = BRICK_DIMS[brick_type]
            if rotation == 0:
                dims_str = f"{brick_l}×{brick_w}"
            else:
                dims_str = f"{brick_w}×{brick_l}"

            print(
                f"      {color}{pattern}{RESET_COLOR} {brick_type} at ({x},{y}) [{dims_str}]"
            )

        print()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
