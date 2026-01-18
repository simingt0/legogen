"""
Test script for Voxelizer module
Run from project root: python3 server/test_voxelizer.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from pipeline.voxelizer import voxelize_mesh


def test_cached_models():
    """Test voxelizing the cached Meshy models"""

    # Check for cached models
    cache_dir = Path(__file__).parent.parent / "pipeline" / "meshy" / "cache"

    if not cache_dir.exists():
        print("❌ No cache directory found")
        print("   Run test_meshy.py first to generate cached models")
        return

    # Find all cached OBJ files
    obj_files = list(cache_dir.glob("*.obj"))

    if not obj_files:
        print("❌ No cached OBJ files found")
        print("   Run test_meshy.py with MESHY_MODE=cache first")
        return

    print("\n" + "=" * 70)
    print("VOXELIZER TEST")
    print("=" * 70)
    print(f"Found {len(obj_files)} cached model(s)")
    print("=" * 70 + "\n")

    # Test each cached model
    for i, obj_path in enumerate(obj_files, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}/{len(obj_files)}: {obj_path.name}")
        print("=" * 70)

        # Try different voxel sizes
        for size in [8, 16, 24]:
            print(f"\n  Testing with size={size}...")

            try:
                voxels = voxelize_mesh(str(obj_path), size=size)

                print(f"  ✅ Success!")
                print(f"     Shape: {voxels.shape}")
                print(f"     Dtype: {voxels.dtype}")
                print(
                    f"     Filled: {voxels.sum():,} / {voxels.size:,} ({100 * voxels.sum() / voxels.size:.1f}%)"
                )

                # Show layer-by-layer stats
                if voxels.ndim == 3:
                    height = voxels.shape[2]
                    print(f"     Layers (z-axis): {height}")
                    for z in range(min(3, height)):
                        layer_filled = voxels[:, :, z].sum()
                        layer_total = voxels[:, :, z].size
                        print(f"       Layer {z}: {layer_filled}/{layer_total} filled")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")


def test_single_file(obj_path: str, size: int = 16):
    """Test voxelizing a single OBJ file"""

    print(f"\nVoxelizing: {obj_path}")
    print(f"Size: {size}")
    print("-" * 60)

    try:
        voxels = voxelize_mesh(obj_path, size=size)

        print(f"\n✅ Success!")
        print(f"   Shape: {voxels.shape}")
        print(f"   Dtype: {voxels.dtype}")
        print(f"   Filled: {voxels.sum():,} / {voxels.size:,}")
        print(f"   Percentage: {100 * voxels.sum() / voxels.size:.2f}%")

        # Visualize top layer
        if voxels.ndim == 3:
            print(f"\n   Top layer (z={voxels.shape[2] - 1}):")
            top_layer = voxels[:, :, -1]
            for y in range(min(8, top_layer.shape[1])):
                row = ""
                for x in range(min(8, top_layer.shape[0])):
                    row += "█" if top_layer[x, y] else "·"
                print(f"   {row}")

        return voxels

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific file
        obj_path = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        test_single_file(obj_path, size)
    else:
        # Test all cached models
        test_cached_models()
