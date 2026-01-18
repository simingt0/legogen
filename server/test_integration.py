"""
Integration test for the full LegoGen pipeline
Tests: Image → Classifier → Bricks
       Description → Meshy → OBJ → Voxelizer → Voxels
       Voxels + Bricks → Builder → Instructions

Run from project root:
    python3 server/test_integration.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.builder import generate_build_instructions
from pipeline.classifier import classify_bricks
from pipeline.meshy import generate_3d_model
from pipeline.voxelizer import voxelize_mesh


async def test_full_pipeline():
    """Test the complete pipeline end-to-end"""

    print("\n" + "=" * 70)
    print("LEGOGEN INTEGRATION TEST")
    print("=" * 70 + "\n")

    # Step 1: Simulate image classification (using stub)
    print("Step 1: Classifying bricks from image...")
    print("-" * 70)

    # Create dummy image bytes
    dummy_image = b"fake image data" * 100
    available_bricks = classify_bricks(dummy_image)

    print(f"✅ Found {sum(available_bricks.values())} bricks:")
    for brick_type, count in sorted(available_bricks.items()):
        if count > 0:
            print(f"   {brick_type}: {count}")

    # Step 2: Generate 3D model from description
    print("\nStep 2: Generating 3D model from description...")
    print("-" * 70)

    test_description = "(low poly, cartoon) a donut"
    print(f"Description: '{test_description}'")

    try:
        # Use test mode to use cached models
        obj_path = await generate_3d_model(
            description=test_description,
            output_dir="/tmp/legogen",
            mode="test",  # Use cached models only
        )
        print(f"✅ Model generated: {obj_path}")

        # Verify file exists
        if Path(obj_path).exists():
            file_size = Path(obj_path).stat().st_size
            print(f"   File size: {file_size:,} bytes")
        else:
            print(f"❌ ERROR: File not found at {obj_path}")
            return

    except RuntimeError as e:
        print(f"❌ Meshy error: {e}")
        print("\nTIP: To cache models, run:")
        print("  MESHY_MODE=cache python3 server/test_meshy.py")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 3: Voxelize the model
    print("\nStep 3: Voxelizing the 3D model...")
    print("-" * 70)

    voxel_size = 16
    try:
        voxels = voxelize_mesh(obj_path, size=voxel_size)
        print(f"✅ Voxelized successfully")
        print(f"   Shape: {voxels.shape}")
        print(f"   Filled: {voxels.sum():,} / {voxels.size:,} voxels")
        print(f"   Percentage: {100 * voxels.sum() / voxels.size:.1f}%")
    except Exception as e:
        print(f"❌ Voxelization error: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 4: Generate build instructions
    print("\nStep 4: Generating build instructions...")
    print("-" * 70)

    max_attempts = 10
    try:
        result = None
        for attempt in range(1, max_attempts + 1):
            print(f"   Attempt {attempt}/{max_attempts}...", end="")

            # Make a copy since builder modifies the dict
            bricks_copy = available_bricks.copy()
            result = generate_build_instructions(voxels, bricks_copy)

            if result["success"]:
                print(" ✅ Success!")
                break
            else:
                error = result.get("error", "Unknown error")
                print(f" ❌ {error}")

        if result and result["success"]:
            print(f"\n✅ Build instructions generated!")
            print(f"   Layers: {len(result['layers'])}")

            total_bricks = sum(len(layer) for layer in result["layers"])
            print(f"   Total bricks: {total_bricks}")

            # Show layer breakdown
            print("\n   Layer breakdown:")
            for z, layer in enumerate(result["layers"]):
                if len(layer) > 0:
                    print(f"     Layer {z}: {len(layer)} bricks")

            # Show brick type usage
            print("\n   Brick usage:")
            brick_counts = {}
            for layer in result["layers"]:
                for brick in layer:
                    brick_type = brick["type"]
                    brick_counts[brick_type] = brick_counts.get(brick_type, 0) + 1

            for brick_type in sorted(brick_counts.keys()):
                count = brick_counts[brick_type]
                print(f"     {brick_type}: {count}")

        else:
            print(f"\n❌ Failed to generate build after {max_attempts} attempts")
            if result:
                print(f"   Final error: {result.get('error', 'Unknown')}")
            return

    except Exception as e:
        print(f"❌ Builder error: {e}")
        import traceback

        traceback.print_exc()
        return

    # Final summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print(f"✅ All pipeline stages completed successfully!")
    print(f"\nFinal output:")
    print(f"  - Voxel grid: {voxels.shape}")
    print(f"  - Layers: {len(result['layers'])}")
    print(f"  - Bricks placed: {total_bricks}")
    print(f"  - Attempts needed: {attempt}")
    print("=" * 70 + "\n")


async def test_multiple_models():
    """Test with multiple cached models"""

    cache_dir = Path(__file__).parent.parent / "pipeline" / "meshy" / "cache"

    if not cache_dir.exists():
        print("❌ No cache directory found")
        print("   Run: MESHY_MODE=cache python3 server/test_meshy.py")
        return

    # Find all cached models
    json_files = list(cache_dir.glob("*.json"))

    if not json_files:
        print("❌ No cached models found")
        print("   Run: MESHY_MODE=cache python3 server/test_meshy.py")
        return

    print("\n" + "=" * 70)
    print(f"TESTING {len(json_files)} CACHED MODELS")
    print("=" * 70 + "\n")

    results = []

    for i, json_file in enumerate(json_files, 1):
        # Load metadata
        with open(json_file, "r") as f:
            metadata = json.load(f)

        description = metadata.get("description", "Unknown")
        obj_file = json_file.with_suffix(".obj")

        if not obj_file.exists():
            print(f"⚠️  Skipping {description}: OBJ file not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"Model {i}/{len(json_files)}: {description}")
        print("=" * 70)

        # Classify bricks (using stub)
        dummy_image = b"fake" * 100
        bricks = classify_bricks(dummy_image)

        # Voxelize
        try:
            voxels = voxelize_mesh(str(obj_file), size=16)
            print(f"✅ Voxelized: {voxels.shape}, {voxels.sum()} filled")
        except Exception as e:
            print(f"❌ Voxelization failed: {e}")
            results.append(
                {"description": description, "success": False, "stage": "voxelize"}
            )
            continue

        # Build
        try:
            result = None
            for attempt in range(1, 11):
                bricks_copy = bricks.copy()
                result = generate_build_instructions(voxels, bricks_copy)
                if result["success"]:
                    break

            if result and result["success"]:
                total = sum(len(layer) for layer in result["layers"])
                print(
                    f"✅ Build succeeded: {len(result['layers'])} layers, {total} bricks"
                )
                results.append(
                    {
                        "description": description,
                        "success": True,
                        "layers": len(result["layers"]),
                        "bricks": total,
                        "attempts": attempt,
                    }
                )
            else:
                print(f"❌ Build failed after 10 attempts")
                results.append(
                    {"description": description, "success": False, "stage": "build"}
                )

        except Exception as e:
            print(f"❌ Build error: {e}")
            results.append(
                {"description": description, "success": False, "stage": "build"}
            )

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-MODEL TEST SUMMARY")
    print("=" * 70)

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    print(f"\nTotal models tested: {len(results)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    if successes:
        print("\n✅ Successful builds:")
        for r in successes:
            print(f"   - {r['description']}")
            print(
                f"     {r['layers']} layers, {r['bricks']} bricks, {r['attempts']} attempts"
            )

    if failures:
        print("\n❌ Failed builds:")
        for r in failures:
            stage = r.get("stage", "unknown")
            print(f"   - {r['description']} (failed at: {stage})")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        asyncio.run(test_multiple_models())
    else:
        asyncio.run(test_full_pipeline())
