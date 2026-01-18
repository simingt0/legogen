"""
Algorithm 4: Floating Cell Priority with Relaxed Orphan Prevention
Prioritizes filling floating cells first, with relaxed rules to avoid orphaning.
"""

import random
from copy import deepcopy

import numpy as np

from .base import BRICK_DIMS, BuilderAlgorithm

# Custom brick priority order (same as Algorithm 2)
CUSTOM_BRICK_ORDER = [
    "2x8",
    "2x6",
    "1x6",
    "2x4",
    "1x4",
    "2x3",
    "1x3",
    "2x2",
    "1x2",
    "1x1",
]


class Algorithm4(BuilderAlgorithm):
    """
    Floating cell priority algorithm with relaxed orphan prevention.

    Strategy:
    - Identify floating cells (cells without vertical neighbors above OR below)
    - Prioritize filling floating cells first
    - Only place bricks on floating cells if:
        * They cover multiple floating cells, OR
        * There are 3 or fewer floating cells remaining
    - For non-floating cells, use random placement
    """

    @property
    def name(self) -> str:
        return "Algorithm 4: Floating Cell Priority (Relaxed)"

    @property
    def description(self) -> str:
        return (
            "Identifies floating cells (no vertical neighbor) and fills them first. "
            "Relaxed rule: allows single floating cell coverage when 3 or fewer remain."
        )

    def build(
        self,
        voxel_grid: np.ndarray,
        available_bricks: dict[str, int],
    ) -> dict:
        """Generate LEGO build instructions prioritizing floating cells."""

        # Validate inputs
        if voxel_grid.size == 0:
            return {"success": False, "layers": None, "error": "Empty voxel grid"}

        if not np.any(voxel_grid):
            return {"success": False, "layers": None, "error": "Empty voxel grid"}

        # Work with a copy of the inventory
        inventory = deepcopy(available_bricks)

        # Get dimensions
        width, depth, height = voxel_grid.shape

        # Count floating cells for each layer to determine processing order
        layer_floating_counts = []
        for z in range(height):
            floating_cells = _identify_floating_cells(voxel_grid, z)
            layer_floating_counts.append((z, len(floating_cells)))

        # Sort layers by number of floating cells (most to least)
        # This ensures we handle the most constrained layers first
        sorted_layers = sorted(layer_floating_counts, key=lambda x: x[1], reverse=True)

        # Store all layers (will be reordered to original sequence at the end)
        layer_results = {}

        # Track the order layers were processed
        layer_order = [z for z, _ in sorted_layers]

        # Process each layer in order of most floating to least
        for z, _ in sorted_layers:
            layer_voxels = voxel_grid[:, :, z]

            # Skip empty layers
            if not np.any(layer_voxels):
                layer_results[z] = []
                continue

            # Identify floating cells for this layer
            floating_cells = _identify_floating_cells(voxel_grid, z)

            # Build this layer
            layer_bricks = []
            layer_claimed = np.zeros((width, depth), dtype=bool)

            # Phase 1: Fill floating cells
            remaining_floating = set(floating_cells)

            while remaining_floating:
                num_floating = len(remaining_floating)

                # Find all placements that cover at least one floating cell
                # Weight by inventory, not by coverage amount
                valid_placements = []

                for brick_type in CUSTOM_BRICK_ORDER:
                    if inventory.get(brick_type, 0) <= 0:
                        continue

                    # Find all placements that cover at least one floating cell
                    for x in range(width):
                        for y in range(depth):
                            for rotation in [0, 90]:
                                if not _can_place_brick(
                                    x,
                                    y,
                                    brick_type,
                                    rotation,
                                    layer_voxels,
                                    layer_claimed,
                                ):
                                    continue

                                # Count how many floating cells this placement covers
                                covered_floating = _count_floating_cells_covered(
                                    x, y, brick_type, rotation, remaining_floating
                                )

                                if covered_floating == 0:
                                    continue  # Doesn't cover any floating cells

                                # Relaxed orphan prevention rule
                                if covered_floating == 1 and num_floating > 3:
                                    continue  # Don't place if only covers 1 and more than 3 remain

                                # Add to valid placements with inventory weight
                                valid_placements.append(
                                    (
                                        x,
                                        y,
                                        brick_type,
                                        rotation,
                                        inventory[brick_type],
                                        covered_floating,
                                    )
                                )

                if not valid_placements:
                    # Can't cover remaining floating cells
                    floating_list = list(remaining_floating)[:5]  # Show first 5
                    inventory_status = ", ".join(
                        f"{bt}: {cnt}"
                        for bt, cnt in sorted(inventory.items())
                        if cnt > 0
                    )
                    return {
                        "success": False,
                        "layers": None,
                        "error": f"Layer {z}: Can't cover {len(remaining_floating)} floating cells (e.g., {floating_list}). Available: {inventory_status}",
                    }

                # Select placement with exponential weighting based on remaining inventory
                # Weight = (remaining_inventory)^2 for exponential preference
                total_inventory = sum(p[4] ** 2 for p in valid_placements)

                if total_inventory == 0:
                    # All have 0 inventory (shouldn't happen), pick randomly
                    x, y, brick_type, rotation, _, covered_floating = random.choice(
                        valid_placements
                    )
                else:
                    # Weighted random selection
                    rand_value = random.uniform(0, total_inventory)
                    cumulative = 0
                    selected = valid_placements[0]

                    for placement in valid_placements:
                        cumulative += (
                            placement[4] ** 2
                        )  # Square for exponential weighting
                        if rand_value <= cumulative:
                            selected = placement
                            break

                    x, y, brick_type, rotation, _, covered_floating = selected

                # Place the brick
                _place_brick(x, y, brick_type, rotation, layer_claimed, layer_bricks)
                inventory[brick_type] -= 1

                # Update remaining floating cells
                covered = _get_covered_cells(x, y, brick_type, rotation)
                for cell in covered:
                    remaining_floating.discard(cell)

            # Phase 2: Fill remaining non-floating cells
            while not _is_layer_complete(layer_voxels, layer_claimed):
                placed = False

                for brick_type in CUSTOM_BRICK_ORDER:
                    if inventory.get(brick_type, 0) <= 0:
                        continue

                    # Find all valid placements
                    valid_placements = []
                    for x in range(width):
                        for y in range(depth):
                            for rotation in [0, 90]:
                                if _can_place_brick(
                                    x,
                                    y,
                                    brick_type,
                                    rotation,
                                    layer_voxels,
                                    layer_claimed,
                                ):
                                    valid_placements.append((x, y, rotation))

                    if valid_placements:
                        # Randomly select one
                        x, y, rotation = random.choice(valid_placements)
                        _place_brick(
                            x, y, brick_type, rotation, layer_claimed, layer_bricks
                        )
                        inventory[brick_type] -= 1
                        placed = True
                        break

                if not placed:
                    # Can't fill remaining cells
                    unfilled = []
                    for x in range(width):
                        for y in range(depth):
                            if layer_voxels[x, y] and not layer_claimed[x, y]:
                                unfilled.append((x, y))

                    inventory_status = ", ".join(
                        f"{bt}: {cnt}"
                        for bt, cnt in sorted(inventory.items())
                        if cnt > 0
                    )
                    unfilled_sample = unfilled[:5]
                    return {
                        "success": False,
                        "layers": None,
                        "error": f"Layer {z}: Can't fill {len(unfilled)} cells (e.g., {unfilled_sample}). Available: {inventory_status}",
                    }

            layer_results[z] = layer_bricks

        # Reorder layers back to original sequence (bottom to top)
        all_layers = [layer_results[z] for z in range(height)]

        # Check connectivity
        if not _check_connectivity(all_layers, voxel_grid.shape):
            return {
                "success": False,
                "layers": None,
                "error": "Structure has floating sections",
            }

        return {
            "success": True,
            "layers": all_layers,
            "error": None,
            "layer_order": layer_order,
        }


def _identify_floating_cells(voxel_grid: np.ndarray, z: int) -> set[tuple[int, int]]:
    """
    Identify floating cells in layer z.
    A cell is floating if it has no neighbor above OR below.

    Special cases:
    - Layer 0: must have neighbor above to not be floating
    - Layer height-1: must have neighbor below to not be floating
    """
    width, depth, height = voxel_grid.shape
    floating = set()

    # Layer 0 is never floating (baseplate)
    if z == 0:
        return floating

    for x in range(width):
        for y in range(depth):
            if not voxel_grid[x, y, z]:
                continue  # Empty cell, skip

            has_neighbor_below = z > 0 and voxel_grid[x, y, z - 1]
            has_neighbor_above = z < height - 1 and voxel_grid[x, y, z + 1]

            # Floating if no neighbor on either side
            if not has_neighbor_below and not has_neighbor_above:
                floating.add((x, y))

    return floating


def _count_floating_cells_covered(
    x: int, y: int, brick_type: str, rotation: int, floating_cells: set[tuple[int, int]]
) -> int:
    """Count how many floating cells this brick placement would cover."""
    covered = _get_covered_cells(x, y, brick_type, rotation)
    return len(covered & floating_cells)


def _get_covered_cells(
    x: int, y: int, brick_type: str, rotation: int
) -> set[tuple[int, int]]:
    """Get set of (x, y) cells covered by this brick placement."""
    brick_width, brick_length = BRICK_DIMS[brick_type]

    if rotation == 0:
        dx, dy = brick_length, brick_width
    else:
        dx, dy = brick_width, brick_length

    covered = set()
    for i in range(dx):
        for j in range(dy):
            covered.add((x + i, y + j))

    return covered


def _can_place_brick(
    x: int,
    y: int,
    brick_type: str,
    rotation: int,
    layer_voxels: np.ndarray,
    layer_claimed: np.ndarray,
) -> bool:
    """Check if a brick can be placed at (x, y) with given rotation."""
    width, depth = layer_voxels.shape
    brick_width, brick_length = BRICK_DIMS[brick_type]

    if rotation == 0:
        dx, dy = brick_length, brick_width
    else:
        dx, dy = brick_width, brick_length

    if x + dx > width or y + dy > depth:
        return False

    for i in range(dx):
        for j in range(dy):
            cell_x, cell_y = x + i, y + j
            if cell_x >= width or cell_y >= depth:
                return False
            if not layer_voxels[cell_x, cell_y]:
                return False
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
    """Place a brick at (x, y) and mark cells as claimed."""
    brick_width, brick_length = BRICK_DIMS[brick_type]

    if rotation == 0:
        dx, dy = brick_length, brick_width
    else:
        dx, dy = brick_width, brick_length

    for i in range(dx):
        for j in range(dy):
            layer_claimed[x + i, y + j] = True

    layer_bricks.append({"type": brick_type, "x": x, "y": y, "rotation": rotation})


def _is_layer_complete(layer_voxels: np.ndarray, layer_claimed: np.ndarray) -> bool:
    """Check if all required cells in the layer are filled."""
    width, depth = layer_voxels.shape

    for x in range(width):
        for y in range(depth):
            if layer_voxels[x, y] and not layer_claimed[x, y]:
                return False

    return True


def _check_connectivity(layers: list[list[dict]], grid_shape: tuple) -> bool:
    """Check if all bricks are connected through vertical layer-to-layer support."""
    if not layers or all(len(layer) == 0 for layer in layers):
        return True

    width, depth, height = grid_shape

    occupied = {}

    for z, layer in enumerate(layers):
        for brick in layer:
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

    for z in range(1, height):
        if len(layers[z]) == 0:
            continue

        layer_has_support = False
        for x in range(width):
            for y in range(depth):
                if occupied.get((z, x, y), False):
                    if occupied.get((z - 1, x, y), False):
                        layer_has_support = True
                        break
            if layer_has_support:
                break

        if not layer_has_support:
            return False

    return True
