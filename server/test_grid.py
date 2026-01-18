"""
Test script for visualizing algorithm behavior on a 2D grid
Run from project root: python3 server/test_grid.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from pipeline.builder import BRICK_DIMS, get_algorithm

# Define test grids
TEST_GRIDS = {
    "grid1": """
[ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][X][X][X][X][ ][ ]
[ ][X][X][X][X][X][ ][ ]
[ ][ ][ ][X][X][X][X][ ]
[ ][ ][ ][X][X][X][X][ ]
[ ][ ][ ][X][X][X][ ][ ]
[ ][ ][ ][X][X][X][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ]
""",
    "grid2": """
[ ][ ][ ][X][X][ ][ ][ ]
[ ][X][X][X][X][X][ ][ ]
[ ][ ][ ][X][X][X][X][ ]
[ ][ ][ ][ ][ ][X][X][X]
[ ][ ][X][X][X][X][X][X]
[ ][ ][X][X][X][X][X][X]
[ ][ ][X][X][X][X][X][ ]
[ ][ ][ ][X][X][X][ ][ ]
[ ][ ][X][X][X][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ]
""",
}


def parse_grid(grid_str: str) -> np.ndarray:
    """Parse ASCII grid into numpy boolean array"""
    lines = [line.strip() for line in grid_str.strip().split("\n") if line.strip()]

    # Parse the grid
    grid_data = []
    for line in lines:
        # Each cell is in format [X] or [ ]
        # Split by ] to get cells
        parts = line.split("]")
        row = []
        for part in parts:
            if "[" in part:
                # Extract what's between the brackets
                content = part.split("[")[-1].strip()
                row.append(content == "X")

        if row:
            grid_data.append(row)

    if not grid_data:
        return np.zeros((1, 1, 1), dtype=bool)

    height = len(grid_data)
    width = max(len(row) for row in grid_data)

    # Convert to numpy array (width, height, 1)
    grid = np.zeros((width, height, 1), dtype=bool)
    for y, row in enumerate(grid_data):
        for x, filled in enumerate(row):
            grid[x, y, 0] = filled

    return grid


def visualize_layer(layer_bricks: list, grid_shape: tuple, layer_voxels: np.ndarray):
    """Visualize the brick placements on a 2D grid"""
    width, height = grid_shape[0], grid_shape[1]

    # Create display grid
    display = [[" " for _ in range(width)] for _ in range(height)]

    # Assign letters to each brick
    brick_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    for i, brick in enumerate(layer_bricks):
        char = brick_chars[i % len(brick_chars)]

        brick_width, brick_length = BRICK_DIMS[brick["type"]]
        if brick["rotation"] == 0:
            dx, dy = brick_length, brick_width
        else:
            dx, dy = brick_width, brick_length

        # Mark cells with this brick's character
        for j in range(dx):
            for k in range(dy):
                x = brick["x"] + j
                y = brick["y"] + k
                if 0 <= x < width and 0 <= y < height:
                    display[y][x] = char

    # Print the grid
    print("\n  " + "─" * (width * 2 + 1))
    for y in range(height):
        row_str = "  │"
        for x in range(width):
            if layer_voxels[x, y]:
                row_str += display[y][x] + "│"
            else:
                row_str += " │"
        print(row_str)
        print("  " + "─" * (width * 2 + 1))

    # Print legend
    print("\n  Brick legend:")
    for i, brick in enumerate(layer_bricks):
        char = brick_chars[i % len(brick_chars)]
        print(
            f"    [{char}] = {brick['type']} at ({brick['x']}, {brick['y']}) rotation={brick['rotation']}°"
        )


def test_grid(
    grid_name: str, grid_array: np.ndarray, algorithm_name: str = "algorithm0"
):
    """Test an algorithm on a 2D grid"""

    print("\n" + "=" * 70)
    print(f"TEST: {grid_name}")
    print("=" * 70)

    # Show original grid
    print("\nOriginal grid:")
    width, height = grid_array.shape[0], grid_array.shape[1]
    filled = grid_array[:, :, 0].sum()

    print(f"  Size: {width} × {height}")
    print(f"  Filled cells: {filled}")

    print("\n  " + "─" * (width * 2 + 1))
    for y in range(height):
        row_str = "  │"
        for x in range(width):
            row_str += ("X" if grid_array[x, y, 0] else " ") + "│"
        print(row_str)
        print("  " + "─" * (width * 2 + 1))

    # Create generous brick inventory
    bricks = {
        "2x6": 20,
        "2x4": 30,
        "2x3": 20,
        "2x2": 30,
        "1x6": 20,
        "1x4": 30,
        "1x3": 20,
        "1x2": 40,
        "1x1": 50,
    }

    # Get algorithm
    print(f"\nAlgorithm: {algorithm_name}")
    try:
        algorithm = get_algorithm(algorithm_name)
        print(f"  {algorithm.name}")
        print(f"  {algorithm.description}")
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Run algorithm multiple times to show variance
    print("\n" + "-" * 70)
    print("Running algorithm (showing first successful attempt)...")
    print("-" * 70)

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}...", end=" ")
        result = algorithm.build(grid_array, bricks.copy())

        if result["success"]:
            print("✅ Success!")

            layer = result["layers"][0]
            print(f"\nBricks used: {len(layer)}")

            # Count brick types
            brick_counts = {}
            for brick in layer:
                brick_type = brick["type"]
                brick_counts[brick_type] = brick_counts.get(brick_type, 0) + 1

            print("\nBrick usage:")
            for brick_type in sorted(brick_counts.keys()):
                count = brick_counts[brick_type]
                area = BRICK_DIMS[brick_type][0] * BRICK_DIMS[brick_type][1]
                print(f"  {brick_type}: {count} (covers {count * area} studs)")

            # Visualize
            print("\nPlacement visualization:")
            visualize_layer(layer, grid_array.shape, grid_array[:, :, 0])
            break
        else:
            print(f"Failed - {result['error']}")

    if not result["success"]:
        print(f"\n❌ Failed after {max_attempts} attempts")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific grid
        grid_name = sys.argv[1]
        algorithm_name = sys.argv[2] if len(sys.argv) > 2 else "algorithm0"

        if grid_name not in TEST_GRIDS:
            print(f"Unknown grid: {grid_name}")
            print(f"Available grids: {', '.join(TEST_GRIDS.keys())}")
            sys.exit(1)

        grid_array = parse_grid(TEST_GRIDS[grid_name])
        test_grid(grid_name, grid_array, algorithm_name)
    else:
        # Test all grids
        print("Usage: python3 server/test_grid.py <grid_name> [algorithm]")
        print(f"\nAvailable grids: {', '.join(TEST_GRIDS.keys())}")
        print("\nExamples:")
        print("  python3 server/test_grid.py cross")
        print("  python3 server/test_grid.py rectangle algorithm0")
        print("\nTesting all grids with algorithm0...")

        for grid_name in TEST_GRIDS.keys():
            grid_array = parse_grid(TEST_GRIDS[grid_name])
            test_grid(grid_name, grid_array, "algorithm0")
            print("\n")
