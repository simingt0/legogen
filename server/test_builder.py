"""
Test script for Builder algorithm
Run from project root: python3 server/test_builder.py <obj_path> <voxel_size> <bricks_json> [algorithm]
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from pipeline.builder import BRICK_DIMS, get_algorithm, list_algorithms
from pipeline.voxelizer import voxelize_mesh

# Color codes for terminal output (bright variants for better visibility)
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
    "\033[90m",  # Gray
    "\033[31;1m",  # Bold Red
    "\033[32;1m",  # Bold Green
    "\033[33;1m",  # Bold Yellow
    "\033[34;1m",  # Bold Blue
    "\033[35;1m",  # Bold Magenta
    "\033[36;1m",  # Bold Cyan
]

# Pattern characters for even more differentiation
BRICK_PATTERNS = [
    "█",
    "▓",
    "▒",
    "░",
    "▪",
    "▫",
    "●",
    "○",
    "◆",
    "◇",
    "■",
    "□",
    "▲",
    "△",
    "▼",
    "▽",
    "★",
    "☆",
]

RESET_COLOR = "\033[0m"


def test_simple_shapes():
    """Test builder with simple geometric shapes"""

    print("\n" + "=" * 70)
    print("BUILDER TEST - SIMPLE SHAPES")
    print("=" * 70 + "\n")

    # Test 1: Single 2x4 brick area
    print("Test 1: Single 2x4 area")
    print("-" * 60)
    voxels = np.ones((2, 4, 1), dtype=bool)
    bricks = {"2x4": 1}
    result = generate_build_instructions(voxels, bricks)
    print(f"Result: {result['success']}")
    if result["success"]:
        print(f"Layers: {len(result['layers'])}")
        print(f"Bricks used: {result['layers']}")
    else:
        print(f"Error: {result['error']}")
    print()

    # Test 2: 4x4 area with various bricks
    print("Test 2: 4x4 solid block")
    print("-" * 60)
    voxels = np.ones((4, 4, 1), dtype=bool)
    bricks = {"2x4": 5, "2x2": 5, "1x1": 10}
    result = generate_build_instructions(voxels, bricks)
    print(f"Result: {result['success']}")
    if result["success"]:
        print(f"Bricks placed: {len(result['layers'][0])}")
        for brick in result["layers"][0]:
            print(f"  {brick}")
    else:
        print(f"Error: {result['error']}")
    print()

    # Test 3: Multi-layer tower
    print("Test 3: 2x2x3 tower")
    print("-" * 60)
    voxels = np.ones((2, 2, 3), dtype=bool)
    bricks = {"2x2": 10, "1x1": 20}
    result = generate_build_instructions(voxels, bricks)
    print(f"Result: {result['success']}")
    if result["success"]:
        for z, layer in enumerate(result["layers"]):
            print(f"  Layer {z}: {len(layer)} bricks")
    else:
        print(f"Error: {result['error']}")
    print()

    # Test 4: Insufficient bricks
    print("Test 4: Insufficient bricks (should fail)")
    print("-" * 60)
    voxels = np.ones((4, 4, 1), dtype=bool)
    bricks = {"2x4": 1}  # Only covers 8 voxels, need 16
    result = generate_build_instructions(voxels, bricks)
    print(f"Result: {result['success']}")
    print(f"Error: {result['error']}")
    print()


def test_with_voxelized_model(
    obj_path: str,
    size: int = 16,
    bricks_json: str = None,
    algorithm_name: str = "algorithm0",
):
    """Test builder with a voxelized 3D model"""

    print("\n" + "=" * 70)
    print(f"BUILDER TEST - VOXELIZED MODEL")
    print("=" * 70)
    print(f"Model: {obj_path}")
    print(f"Voxel size: {size}")
    if bricks_json:
        print(f"Bricks file: {bricks_json}")
    print(f"Algorithm: {algorithm_name}")
    print("-" * 70 + "\n")

    # Voxelize the model
    print("Step 1: Voxelizing model...")
    try:
        voxels = voxelize_mesh(obj_path, size=size)
        print(f"✅ Voxelized: {voxels.shape}")
        print(f"   Filled voxels: {voxels.sum():,}")
    except Exception as e:
        print(f"❌ Voxelization failed: {e}")
        return

    # Load brick inventory
    print("\nStep 2: Preparing brick inventory...")
    if bricks_json:
        try:
            with open(bricks_json, "r") as f:
                bricks = json.load(f)
            print(f"   Loaded from {bricks_json}")
        except Exception as e:
            print(f"❌ Failed to load bricks JSON: {e}")
            return
    else:
        # Default inventory
        bricks = {
            "2x6": 50,
            "2x4": 100,
            "2x3": 80,
            "2x2": 100,
            "1x6": 50,
            "1x4": 100,
            "1x3": 80,
            "1x2": 150,
            "1x1": 200,
        }
        print(f"   Using default inventory")
    total_brick_area = sum(
        BRICK_DIMS[bt][0] * BRICK_DIMS[bt][1] * count for bt, count in bricks.items()
    )
    print(f"   Total brick coverage: {total_brick_area:,} studs")
    print(f"   Voxels to cover: {voxels.sum():,}")

    # Get the algorithm
    print(f"\nStep 3: Loading algorithm '{algorithm_name}'...")
    try:
        algorithm = get_algorithm(algorithm_name)
        print(f"   {algorithm.name}")
        print(f"   {algorithm.description}")
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Generate build instructions with retry
    print("\nStep 4: Generating build instructions...")
    max_attempts = 64

    try:
        result = None
        for attempt in range(1, max_attempts + 1):
            print(f"Try #{attempt}...", end="\r")
            result = algorithm.build(voxels, bricks.copy())

            if result["success"]:
                print(f"✅ Success on attempt {attempt}!")

                # Show layer processing order if available
                if "layer_order" in result:
                    order_str = "→".join(str(z) for z in result["layer_order"])
                    print(f"   Layer processing order: {order_str}")

                break

        if result and result["success"]:
            print(
                f"\n✅ Build instructions generated! (succeeded on attempt {attempt})"
            )
            print(f"\n   Layers: {len(result['layers'])}")

            total_bricks = sum(len(layer) for layer in result["layers"])
            print(f"   Total bricks: {total_bricks}")

            # Show layer breakdown
            print("\n   Layer breakdown:")
            for z, layer in enumerate(result["layers"]):
                if len(layer) > 0:
                    print(f"     Layer {z}: {len(layer)} bricks")

            # Detailed layer-by-layer instructions with visual grid
            print("\n" + "=" * 70)
            print("DETAILED LAYER-BY-LAYER INSTRUCTIONS")
            print("=" * 70)

            width, depth = voxels.shape[0], voxels.shape[1]

            for z, layer in enumerate(result["layers"]):
                if len(layer) == 0:
                    continue

                print(f"\n>>> LAYER {z} <<<")
                print(f"    Bricks: {len(layer)}")

                # Create visual grid for this layer with colors
                display = [[-1 for _ in range(width)] for _ in range(depth)]
                color_map = {}

                # Assign colors and patterns to each brick
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

                # Print visual grid with colors
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

                # Print legend with colors and patterns
                print("\n    Legend:")
                for i, brick in enumerate(layer):
                    color, pattern = color_map[i]
                    brick_type = brick["type"]
                    x, y = brick["x"], brick["y"]
                    rotation = brick["rotation"]

                    brick_w, brick_l = BRICK_DIMS[brick_type]
                    if rotation == 0:
                        dims_str = f"{brick_l}×{brick_w} (horizontal)"
                    else:
                        dims_str = f"{brick_w}×{brick_l} (vertical)"

                    print(
                        f"      {color}{pattern}{RESET_COLOR} {brick_type} at ({x},{y}) - {dims_str}"
                    )

            # Show brick type usage
            print("\n   Brick type usage:")
            brick_counts = {}
            for layer in result["layers"]:
                for brick in layer:
                    brick_type = brick["type"]
                    brick_counts[brick_type] = brick_counts.get(brick_type, 0) + 1

            for brick_type in sorted(brick_counts.keys()):
                count = brick_counts[brick_type]
                print(f"     {brick_type}: {count}")

        else:
            print(f"\n❌ Failed after {max_attempts} attempts")
            if result:
                print(f"   Error: {result['error']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def test_cached_models():
    """Test builder with all cached Meshy models"""

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
    print("BUILDER TEST - ALL CACHED MODELS")
    print("=" * 70)
    print(f"Found {len(obj_files)} cached model(s)")
    print("=" * 70 + "\n")

    for i, obj_path in enumerate(obj_files, 1):
        print(f"\n{'=' * 70}")
        print(f"Model {i}/{len(obj_files)}: {obj_path.stem}")
        print("=" * 70)

        test_with_voxelized_model(str(obj_path), size=16)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "simple":
            # Test simple shapes
            test_simple_shapes()
        elif sys.argv[1] == "cached":
            # Test all cached models
            test_cached_models()
        elif sys.argv[1] == "list":
            # List available algorithms
            print("\nAvailable algorithms:")
            print("=" * 70)
            for name, desc in list_algorithms().items():
                print(f"\n{name}:")
                print(f"  {desc}")
            print("\n" + "=" * 70)
        else:
            # Test specific OBJ file with arguments
            obj_path = sys.argv[1]
            size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
            bricks_json = sys.argv[3] if len(sys.argv) > 3 else None
            algorithm_name = sys.argv[4] if len(sys.argv) > 4 else "algorithm0"

            if not Path(obj_path).exists():
                print(f"Error: File not found: {obj_path}")
                sys.exit(1)

            if bricks_json and not Path(bricks_json).exists():
                print(f"Error: Bricks file not found: {bricks_json}")
                sys.exit(1)

            test_with_voxelized_model(obj_path, size, bricks_json, algorithm_name)
    else:
        # Default: show usage
        print("Usage:")
        print("  python3 server/test_builder.py simple")
        print("  python3 server/test_builder.py cached")
        print("  python3 server/test_builder.py list")
        print(
            "  python3 server/test_builder.py <obj_path> <voxel_size> [bricks_json] [algorithm]"
        )
        print()
        print("Examples:")
        print(
            "  python3 server/test_builder.py ~/flower2.obj 16 pipeline/builder/bricks1.json algorithm0"
        )
        print("  python3 server/test_builder.py ~/lizard.obj 32")
        print("  python3 server/test_builder.py list")
        print()
        sys.exit(1)
