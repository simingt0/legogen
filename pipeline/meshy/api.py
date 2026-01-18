"""
Meshy module - generates 3D models from text descriptions
See plan.md for full specification
"""

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
MESHY_API_KEY = os.environ.get("MESHY_API_KEY")
MESHY_BASE_URL = "https://api.meshy.ai"

# Cache configuration
CACHE_DIR = Path(__file__).parent / "cache"
MESHY_MODE = os.environ.get("MESHY_MODE", "test").lower()  # test, cache, prod, or mock


async def generate_3d_model(
    description: str,
    output_dir: str = "/tmp/legogen",
    art_style: str = "sculpture",
    timeout: int = 300,
    mode: str = None,
) -> str:
    """
    Generate a 3D model from a text description using Meshy API.

    Args:
        description: Text description of what to create (max 600 chars)
        output_dir: Directory to save the downloaded OBJ file
        art_style: "sculpture" (blocky, better for voxelization) or "realistic"
        timeout: Max seconds to wait for generation (default 5 minutes)
        mode: Override mode - "test" (use cache only), "cache" (query and cache),
              "prod" (use cache if available, else query and cache), "mock" (generate simple test OBJ)
              If None, uses MESHY_MODE environment variable (default: "test")

    Returns:
        Path to the downloaded OBJ file (e.g., "/tmp/legogen/abc123.obj")

    Raises:
        ValueError: If description is empty or too long
        TimeoutError: If generation takes longer than timeout
        RuntimeError: If API request fails or returns error
    """
    # Validate inputs
    if not description or not description.strip():
        raise ValueError("Description cannot be empty")

    if len(description) > 600:
        raise ValueError("Description too long (max 600 chars)")

    # Determine mode
    active_mode = mode if mode is not None else MESHY_MODE

    if active_mode not in ["test", "cache", "prod", "mock"]:
        raise ValueError(
            f"Invalid mode: {active_mode}. Must be 'test', 'cache', 'prod', or 'mock'"
        )

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # MOCK MODE: Generate simple test OBJ file
    if active_mode == "mock":
        return _generate_mock_obj(description, output_dir)

    # TEST MODE: Use cached response only
    if active_mode == "test":
        cached_path = _get_cached_model(description, art_style, output_dir)
        if cached_path:
            print(f"[TEST MODE] Using cached model for '{description}'")
            return cached_path
        else:
            raise RuntimeError(
                f"[TEST MODE] No cached model found for '{description}'. "
                f"Run with MESHY_MODE=cache to generate and cache this model."
            )

    # PROD MODE: Try cache first, then query if needed
    if active_mode == "prod":
        cached_path = _get_cached_model(description, art_style, output_dir)
        if cached_path:
            print(f"[PROD MODE] Using cached model for '{description}'")
            return cached_path
        else:
            print(f"[PROD MODE] No cache found for '{description}', querying API...")

    # CACHE or PROD MODE (no cache hit): Query the API
    if not MESHY_API_KEY:
        raise RuntimeError("MESHY_API_KEY environment variable not set")

    # HTTP client with timeout
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create preview task
        task_id = await _create_preview_task(client, description)

        # Step 2: Poll for completion
        model_url = await _poll_for_completion(client, task_id, timeout)

        # Step 3: Download OBJ file
        obj_path = await _download_obj(client, task_id, model_url, output_dir)

        # CACHE or PROD MODE: Save to cache
        if active_mode in ["cache", "prod"]:
            _cache_model(description, art_style, obj_path)
            mode_label = "CACHE" if active_mode == "cache" else "PROD"
            print(f"[{mode_label} MODE] Model generated and cached")

        return obj_path


async def _create_preview_task(
    client: httpx.AsyncClient,
    description: str,
) -> str:
    """
    Create a text-to-3D preview task.

    Returns:
        task_id: The ID of the created task
    """
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "mode": "preview",
        "prompt": description,
        "art_style": "sculpture",
        "ai_model": "meshy-4",
        "target_polycount": 10000,
    }

    try:
        response = await client.post(
            f"{MESHY_BASE_URL}/v2/text-to-3d",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        if "result" not in data:
            raise RuntimeError(f"Unexpected API response: {data}")

        return data["result"]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise RuntimeError("Meshy API authentication failed - check API key")
        elif e.response.status_code == 400:
            raise RuntimeError(f"Bad request to Meshy API: {e.response.text}")
        else:
            raise RuntimeError(
                f"Meshy API error: {e.response.status_code} - {e.response.text}"
            )
    except httpx.RequestError as e:
        raise RuntimeError(f"Network error connecting to Meshy API: {e}")


async def _poll_for_completion(
    client: httpx.AsyncClient,
    task_id: str,
    timeout: int,
) -> str:
    """
    Poll the task until it completes or times out.

    Returns:
        model_url: URL to download the OBJ file
    """
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
    }

    start_time = asyncio.get_event_loop().time()
    poll_interval = 5  # seconds
    last_poll_time = start_time

    while True:
        # Check timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Model generation timed out after {timeout}s")

        try:
            response = await client.get(
                f"{MESHY_BASE_URL}/v2/text-to-3d/{task_id}",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status")
            progress = data.get("progress", 0)

            # Calculate timing
            current_time = asyncio.get_event_loop().time()
            elapsed_total = current_time - start_time
            time_since_last = current_time - last_poll_time
            last_poll_time = current_time

            # Log progress only on significant updates (every 20%)
            if progress % 20 == 0 or progress > 90:
                print(f"   Meshy: {progress}% complete ({elapsed_total:.0f}s elapsed)")

            if status == "SUCCEEDED":
                model_urls = data.get("model_urls", {})
                obj_url = model_urls.get("obj")

                if not obj_url:
                    raise RuntimeError("No OBJ URL in completed task response")

                return obj_url

            elif status == "FAILED":
                raise RuntimeError(
                    f"Meshy API task failed: {data.get('error', 'Unknown error')}"
                )

            elif status in ["PENDING", "IN_PROGRESS"]:
                # Wait before next poll
                await asyncio.sleep(poll_interval)

            else:
                raise RuntimeError(f"Unknown task status: {status}")

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Error polling Meshy task: {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error polling Meshy API: {e}")


async def _download_obj(
    client: httpx.AsyncClient,
    task_id: str,
    model_url: str,
    output_dir: str,
) -> str:
    """
    Download the OBJ file from the provided URL.

    Returns:
        Path to the saved OBJ file
    """
    output_path = Path(output_dir) / f"{task_id}.obj"

    try:
        response = await client.get(model_url)
        response.raise_for_status()

        # Write to file
        with open(output_path, "wb") as f:
            f.write(response.content)

        return str(output_path)

    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Failed to download model file: {e.response.status_code}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Network error downloading model: {e}")
    except IOError as e:
        raise RuntimeError(f"Failed to save model file: {e}")


def _get_cache_key(description: str, art_style: str) -> str:
    """
    Generate a cache key from description and art style.
    """
    content = f"{description}|{art_style}"
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached_model(description: str, art_style: str, output_dir: str) -> str | None:
    """
    Check if a cached model exists for this description.

    Returns:
        Path to cached model if exists, None otherwise
    """
    if not CACHE_DIR.exists():
        return None

    cache_key = _get_cache_key(description, art_style)
    cache_meta_path = CACHE_DIR / f"{cache_key}.json"
    cache_obj_path = CACHE_DIR / f"{cache_key}.obj"

    if not cache_meta_path.exists() or not cache_obj_path.exists():
        return None

    # Read metadata
    try:
        with open(cache_meta_path, "r") as f:
            metadata = json.load(f)

        # Verify the metadata matches
        if (
            metadata.get("description") != description
            or metadata.get("art_style") != art_style
        ):
            return None

        # Copy cached file to output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use the original task_id from cache metadata as filename
        task_id = metadata.get("task_id", cache_key)
        output_file = output_path / f"{task_id}.obj"

        shutil.copy2(cache_obj_path, output_file)
        print(f"   Using cached model for '{description}'")
        return str(output_file)

    except (json.JSONDecodeError, IOError):
        return None


def _cache_model(description: str, art_style: str, obj_path: str) -> None:
    """
    Cache a generated model for future use.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = _get_cache_key(description, art_style)
    cache_meta_path = CACHE_DIR / f"{cache_key}.json"
    cache_obj_path = CACHE_DIR / f"{cache_key}.obj"

    try:
        # Extract task_id from the obj_path filename
        task_id = Path(obj_path).stem

        # Save metadata
        metadata = {
            "description": description,
            "art_style": art_style,
            "task_id": task_id,
            "original_path": obj_path,
        }

        with open(cache_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Copy OBJ file to cache
        shutil.copy2(obj_path, cache_obj_path)

    except IOError as e:
        print(f"Warning: Failed to cache model: {e}")


def _generate_mock_obj(description: str, output_dir: str) -> str:
    """
    Generate a simple mock OBJ file for testing without API calls.

    Returns:
        Path to the generated OBJ file
    """
    import time

    # Generate a simple cube OBJ
    obj_content = """# Mock OBJ file for testing
# Description: {description}
# Generated: {timestamp}

# Vertices
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0

# Faces
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
""".format(description=description, timestamp=time.time())

    # Create unique filename based on description
    cache_key = _get_cache_key(description, "sculpture")
    output_path = Path(output_dir) / f"mock_{cache_key}.obj"

    # Write the file
    with open(output_path, "w") as f:
        f.write(obj_content)

    return str(output_path)
