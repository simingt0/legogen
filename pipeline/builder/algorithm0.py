"""
Algorithm 0: Greedy Position-by-Position with Monte Carlo
Shuffles unfilled positions and places largest available brick at each position.
"""

import random
from collections import deque
from copy import deepcopy

import numpy as np

from .base import BRICK_DIMS, BRICKS_BY_SIZE, BuilderAlgorithm


class Algorithm0(BuilderAlgorithm):
    """
    Greedy position-by-position algorithm with Monte Carlo randomization.

    Strategy:
    - Process each layer from bottom to top
    - Shuffle unfilled positions randomly (Monte Carlo variance)
    - For each position, try to place the highest-scoring brick
    - Scoring: area^2 * inventory_weight * random_factor
    - Heavy penalty for 1x1 bricks
    """

    @property
    def name(self) -> str:
        return "Algorithm 0: Greedy Position-by-Position"

    @property
    def description(self) -> str:
        return (
            "Shuffles unfilled positions and greedily places the largest "
            "available brick at each position. Uses Monte Carlo randomization "
            "for retry variance."
        )

    def build(
        self,
        voxel_grid: np.ndarray,
        available_bricks: dict[str, int],
    ) -> dict:
        """Generate LEGO build instructions using greedy position-by-position approach."""

        # Validate inputs
        if voxel_grid.size == 0:
            return {"success": False, "layers": None, "error": "Empty voxel grid"}

        if not np.any(voxel_grid):
            return {"success": False, "layers": None, "error": "Empty voxel grid"}

        # Work with a copy of the inventory
        inventory = deepcopy(available_bricks)

        # Get dimensions
        width, depth, height = voxel_grid.shape

        # Store all layers
        all_layers = []

        # Track which voxels have been assigned to bricks
        claimed = np.zeros_like(voxel_grid, dtype=bool)

        # Process each layer from bottom to top
        for z in range(height):
            layer_voxels = voxel_grid[:, :, z]

            # Skip empty layers
            if not np.any(layer_voxels):
                all_layers.append([])
                continue

            # Build this layer
            layer_bricks = []
            layer_claimed = np.zeros((width, depth), dtype=bool)

            # Get list of unfilled positions in random order (Monte Carlo variance)
            unfilled_positions = [(x, y) for x in range(width) for y in range(depth)]
            random.shuffle(unfilled_positions)

            # Try to fill each unfilled position
            for x, y in unfilled_positions:
                # Skip if already claimed or not needed
                if layer_claimed[x, y] or not layer_voxels[x, y]:
                    continue

                # Try to place a brick here
                brick_placed = _try_place_brick(
                    x, y, layer_voxels, layer_claimed, inventory, layer_bricks
                )

                if not brick_placed:
                    # Should not happen if we have 1x1 bricks available
                    inventory_status = ", ".join(
                        f"{bt}: {cnt}" for bt, cnt in sorted(inventory.items())
                    )
                    return {
                        "success": False,
                        "layers": None,
                        "error": f"Cannot fill position ({x}, {y}, {z}) - insufficient bricks. Remaining: {inventory_status}",
                    }

            # Mark claimed voxels in 3D grid
            claimed[:, :, z] = layer_claimed
            all_layers.append(layer_bricks)

        # Check connectivity
        if not _check_connectivity(all_layers, voxel_grid.shape):
            return {
                "success": False,
                "layers": None,
                "error": "Structure has floating sections",
            }

        return {"success": True, "layers": all_layers, "error": None}


def _try_place_brick(
    x: int,
    y: int,
    layer_voxels: np.ndarray,
    layer_claimed: np.ndarray,
    inventory: dict[str, int],
    layer_bricks: list,
) -> bool:
    """
    Try to place the best available brick at position (x, y).

    Returns:
        True if brick was placed, False otherwise
    """
    width, depth = layer_voxels.shape

    # Create weighted list of bricks to try
    # Prioritize: (1) larger bricks, (2) bricks with more inventory
    brick_candidates = []

    for brick_type in BRICKS_BY_SIZE:
        if inventory.get(brick_type, 0) <= 0:
            continue

        # Calculate score: area * log(inventory + 1) * random factor
        brick_width, brick_length = BRICK_DIMS[brick_type]
        area = brick_width * brick_length
        inv_count = inventory[brick_type]

        # Base score - square the area to heavily favor larger bricks
        score = (area**2) * (1 + 0.2 * np.log(inv_count + 1))

        # Heavy penalty for 1x1 bricks - only use as last resort
        if brick_type == "1x1":
            score *= 0.05

        # Monte Carlo randomization
        score *= random.uniform(0.8, 1.2)

        brick_candidates.append((score, brick_type))

    # Sort by score descending
    brick_candidates.sort(reverse=True, key=lambda x: x[0])

    # Try each brick candidate
    for _, brick_type in brick_candidates:
        # Try both rotations
        for rotation in [0, 90]:
            if _can_place_brick(
                x, y, brick_type, rotation, layer_voxels, layer_claimed
            ):
                # Place the brick
                _place_brick(x, y, brick_type, rotation, layer_claimed, layer_bricks)

                # Update inventory
                inventory[brick_type] -= 1

                return True

    return False


def _can_place_brick(
    x: int,
    y: int,
    brick_type: str,
    rotation: int,
    layer_voxels: np.ndarray,
    layer_claimed: np.ndarray,
) -> bool:
    """
    Check if a brick can be placed at (x, y) with given rotation.
    """
    width, depth = layer_voxels.shape
    brick_width, brick_length = BRICK_DIMS[brick_type]

    # Determine actual dimensions based on rotation
    if rotation == 0:
        # Length along X, width along Y
        dx, dy = brick_length, brick_width
    else:  # rotation == 90
        # Length along Y, width along X
        dx, dy = brick_width, brick_length

    # Check bounds
    if x + dx > width or y + dy > depth:
        return False

    # Check all cells that would be covered
    for i in range(dx):
        for j in range(dy):
            cell_x, cell_y = x + i, y + j

            # Must be within bounds (already checked, but defensive)
            if cell_x >= width or cell_y >= depth:
                return False

            # Must need to be filled (True in voxel grid)
            if not layer_voxels[cell_x, cell_y]:
                return False

            # Must not already be claimed
            if layer_claimed[cell_x, cell_y]:
                return False

    return True


def _place_brick(
    x: int,
    y: int,
    brick_type: str,
    rotation: int,
    layer_claimed: np.ndarray,
    layer_bricks: list,
) -> None:
    """
    Place a brick at (x, y) and mark cells as claimed.
    """
    brick_width, brick_length = BRICK_DIMS[brick_type]

    # Determine actual dimensions based on rotation
    if rotation == 0:
        dx, dy = brick_length, brick_width
    else:
        dx, dy = brick_width, brick_length

    # Mark cells as claimed
    for i in range(dx):
        for j in range(dy):
            layer_claimed[x + i, y + j] = True

    # Record brick placement
    layer_bricks.append({"type": brick_type, "x": x, "y": y, "rotation": rotation})


def _check_connectivity(layers: list[list[dict]], grid_shape: tuple) -> bool:
    """
    Check if all bricks are connected through vertical layer-to-layer support.

    Rules:
    - Layer 0 is always supported (on the ground)
    - Each layer above 0 must have at least one occupied cell that connects to the layer below
    - Bricks within a layer can connect horizontally to each other
    - This allows overhangs and complex shapes as long as each layer is anchored
    """
    if not layers or all(len(layer) == 0 for layer in layers):
        return True

    width, depth, height = grid_shape

    # Build a map of which cells are occupied in each layer
    occupied = {}  # (layer, x, y) -> True if occupied

    for z, layer in enumerate(layers):
        for brick in layer:
            # Get cells covered by this brick
            brick_width, brick_length = BRICK_DIMS[brick["type"]]
            if brick["rotation"] == 0:
                dx, dy = brick_length, brick_width
            else:
                dx, dy = brick_width, brick_length

            for i in range(dx):
                for j in range(dy):
                    cell_x = brick["x"] + i
                    cell_y = brick["y"] + j
                    occupied[(z, cell_x, cell_y)] = True

    # Check each layer (except layer 0) has connection to layer below
    for z in range(1, height):
        if len(layers[z]) == 0:
            continue

        # Check if ANY cell in this layer has a cell directly below it
        layer_has_support = False
        for x in range(width):
            for y in range(depth):
                if occupied.get((z, x, y), False):
                    # Check if there's a cell directly below
                    if occupied.get((z - 1, x, y), False):
                        layer_has_support = True
                        break
            if layer_has_support:
                break

        # If this layer has no vertical connection to the layer below, it's floating
        if not layer_has_support:
            return False

    return True


# Wrapper function for backward compatibility
def generate_build_instructions(
    voxel_grid: np.ndarray,
    available_bricks: dict[str, int],
) -> dict:
    """
    Generate LEGO build instructions from a voxel grid using Algorithm 7.

    This is a wrapper function for backward compatibility.
    Uses Algorithm 9 (Floating Component Bonus) by default.
    """
    from .algorithm9 import Algorithm9

    algorithm = Algorithm9()
    return algorithm.build(voxel_grid, available_bricks)
