"""
Algorithm 1: Exhaustive Piece-by-Piece with Random Placement
Tries to place each brick type exhaustively before moving to smaller pieces.
"""

import random
from copy import deepcopy

import numpy as np

from .base import BRICK_DIMS, BRICKS_BY_SIZE, BuilderAlgorithm


class Algorithm1(BuilderAlgorithm):
    """
    Exhaustive piece-by-piece algorithm with random placement.

    Strategy:
    - Process each layer from bottom to top
    - For each brick type (largest to smallest):
        - Find ALL valid placements for this brick type
        - Randomly select one and place it
        - Repeat until no more valid placements exist
    - Move to next smaller brick type
    - If layer cannot be completely filled, fail
    """

    @property
    def name(self) -> str:
        return "Algorithm 1: Exhaustive Piece-by-Piece"

    @property
    def description(self) -> str:
        return (
            "For each brick type (largest to smallest), exhaustively finds all valid "
            "placements and randomly places bricks until no more fit. Pure random selection."
        )

    def build(
        self,
        voxel_grid: np.ndarray,
        available_bricks: dict[str, int],
    ) -> dict:
        """Generate LEGO build instructions using exhaustive piece-by-piece approach."""

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

            # Try each brick type from largest to smallest
            for brick_type in BRICKS_BY_SIZE:
                # Skip if no inventory
                if inventory.get(brick_type, 0) <= 0:
                    continue

                # Exhaust this brick type
                while inventory[brick_type] > 0:
                    # Find all valid placements for this brick type
                    valid_placements = _find_all_valid_placements(
                        brick_type, layer_voxels, layer_claimed, width, depth
                    )

                    if not valid_placements:
                        # No more places for this brick type
                        break

                    # Randomly select one placement
                    placement = random.choice(valid_placements)
                    x, y, rotation = placement

                    # Place the brick
                    _place_brick(
                        x, y, brick_type, rotation, layer_claimed, layer_bricks
                    )

                    # Update inventory
                    inventory[brick_type] -= 1

            # Check if layer is completely filled
            if not _is_layer_complete(layer_voxels, layer_claimed):
                # Get unfilled cells
                unfilled = []
                for x in range(width):
                    for y in range(depth):
                        if layer_voxels[x, y] and not layer_claimed[x, y]:
                            unfilled.append((x, y))

                inventory_status = ", ".join(
                    f"{bt}: {cnt}" for bt, cnt in sorted(inventory.items())
                )
                return {
                    "success": False,
                    "layers": None,
                    "error": f"Cannot fill layer {z}. Unfilled cells: {len(unfilled)}. Remaining: {inventory_status}",
                }

            all_layers.append(layer_bricks)

        # Check connectivity
        if not _check_connectivity(all_layers, voxel_grid.shape):
            return {
                "success": False,
                "layers": None,
                "error": "Structure has floating sections",
            }

        return {"success": True, "layers": all_layers, "error": None}


def _find_all_valid_placements(
    brick_type: str,
    layer_voxels: np.ndarray,
    layer_claimed: np.ndarray,
    width: int,
    depth: int,
) -> list[tuple[int, int, int]]:
    """
    Find all valid placements for a brick type in the current layer.

    Returns:
        List of (x, y, rotation) tuples for valid placements
    """
    valid_placements = []

    # Try every position and rotation
    for x in range(width):
        for y in range(depth):
            for rotation in [0, 90]:
                if _can_place_brick(
                    x, y, brick_type, rotation, layer_voxels, layer_claimed
                ):
                    valid_placements.append((x, y, rotation))

    return valid_placements


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

            # Must be within bounds
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


def _is_layer_complete(layer_voxels: np.ndarray, layer_claimed: np.ndarray) -> bool:
    """
    Check if all required cells in the layer are filled.
    """
    width, depth = layer_voxels.shape

    for x in range(width):
        for y in range(depth):
            # If this cell needs to be filled but isn't claimed, layer is incomplete
            if layer_voxels[x, y] and not layer_claimed[x, y]:
                return False

    return True


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
