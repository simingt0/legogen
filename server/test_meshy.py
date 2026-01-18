"""
Test script for Meshy API integration
Run from project root: python3 server/test_meshy.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.meshy import generate_3d_model


async def test_simple_models():
    """Test generating a few simple 3D models"""

    test_descriptions = [
        "(low poly, cartoon) a flower",
        "(low poly, cartoon) a lizard",
        "(low poly, cartoon) a small house with a chimney",
    ]

    print("\n" + "=" * 70)
    print("MESHY API TEST")
    print("=" * 70)
    print(f"Mode: {os.environ.get('MESHY_MODE', 'test')}")
    print(f"API Key set: {'Yes' if os.environ.get('MESHY_API_KEY') else 'No'}")
    print("=" * 70 + "\n")

    for i, description in enumerate(test_descriptions, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}/{len(test_descriptions)}: '{description}'")
        print("=" * 70)

        try:
            obj_path = await generate_3d_model(
                description=description,
                output_dir="/tmp/legogen",
                timeout=300,  # 5 minutes
            )

            print(f"\n✅ SUCCESS!")
            print(f"   Model saved to: {obj_path}")

            # Verify file exists and show stats
            if os.path.exists(obj_path):
                file_size = os.path.getsize(obj_path)
                print(f"   File size: {file_size:,} bytes")

                # Count lines to give sense of model complexity
                with open(obj_path, "r") as f:
                    lines = sum(1 for _ in f)
                print(f"   Lines in OBJ: {lines:,}")
            else:
                print(f"   ⚠️  Warning: Path returned but file not found")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")


async def test_single_model(description: str):
    """Test a single model generation"""
    print(f"\nGenerating model for: '{description}'")

    try:
        obj_path = await generate_3d_model(
            description=description, output_dir="/tmp/legogen", timeout=300
        )

        print(f"✅ Success! Model: {obj_path}")
        return obj_path

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    # Check if custom description provided
    if len(sys.argv) > 1:
        # Single test with custom description
        custom_description = " ".join(sys.argv[1:])
        asyncio.run(test_single_model(custom_description))
    else:
        # Run all test cases
        asyncio.run(test_simple_models())
