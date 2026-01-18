"""
Meshy module - generates 3D models from text descriptions
See plan.md for full specification
"""
import os


# API key from environment
MESHY_API_KEY = os.environ.get("MESHY_API_KEY")
MESHY_BASE_URL = "https://api.meshy.ai"


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
        art_style: "sculpture" (blocky) or "realistic"
        timeout: Max seconds to wait for generation

    Returns:
        Path to the downloaded OBJ file
    """
    raise NotImplementedError("See plan.md for implementation details")
