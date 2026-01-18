"""
Algorithm 5: Score-Based Placement Evaluation
Evaluates multiple placement options per brick and selects highest scoring placement.
"""

import random
from copy import deepcopy

import numpy as np

from .base import BRICK_DIMS, BuilderAlgorithm

# Custom brick priority order
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


class Algorithm5(BuilderAlgorithm):
    """
    Score-based placement evaluation algorithm.

    Strategy:
    - For each brick type, evaluate up to N placements (N = brick area)
    - Score each placement based on:
        * +5 per floating cell covered
        * +2 for not overlapping exactly with layer below
        * +1 per stud squared in the brick
        * +1 for each brick type with less remaining inventory
    - Place highest scoring brick
    - Track least recently used brick type
    """

    @property
    def name(self) -> str:
        return "Algorithm 5: Score-Based Placement"

    @property
    def description(self) -> str:
        return (
            "Evaluates multiple placements per brick using sophisticated scoring: "
            "floating cells (+5), non-overlap (+2), studs² (+1 to +256), "
            "inventory ranking (+0 to +N-1)."
        )

    def build(
        self,
        voxel_grid: np.ndarray,
        available_bricks: dict[str, int],
    ) -> dict:
        """Generate LEGO build instructions using score-based evaluation."""

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

        # Track least recently used brick type
        last_used_brick = None

        # Process each layer from bottom to top
        for z in range(height):
            layer_voxels = voxel_grid[:, :, z]

            # Skip empty layers
            if not np.any(layer_voxels):
                all_layers.append([])
                continue

            # Identify floating cells for this layer
            floating_cells = _identify_floating_cells(voxel_grid, z)

            # Get layer below for overlap detection
            layer_below = None
            if z > 0:
                layer_below = all_layers[z - 1]

            # Build this layer
            layer_bricks = []
            layer_claimed = np.zeros((width, depth), dtype=bool)

            # Continue placing bricks until layer is complete
            while not _is_layer_complete(layer_voxels, layer_claimed):
                # Evaluate all possible placements and pick the best
                best_placement = None
                best_score = -float("inf")

                for brick_type in CUSTOM_BRICK_ORDER:
                    if inventory.get(brick_type, 0) <= 0:
                        continue

                    # Calculate max placements to evaluate (up to brick area)
                    brick_area = BRICK_DIMS[brick_type][0] * BRICK_DIMS[brick_type][1]

                    # Find all valid placements
                    all_placements = _find_all_valid_placements(
                        brick_type, layer_voxels, layer_claimed, width, depth
                    )

                    if not all_placements:
                        continue

                    # Randomly sample up to brick_area placements
                    placements_to_evaluate = random.sample(
                        all_placements, min(brick_area, len(all_placements))
                    )

                    # Score each placement
                    for x, y, rotation in placements_to_evaluate:
                        score = _score_placement(
                            x,
                            y,
                            brick_type,
                            rotation,
                            floating_cells,
                            layer_below,
                            inventory,
                            last_used_brick,
                        )

                        if score > best_score:
                            best_score = score
                            best_placement = (x, y, brick_type, rotation)

                if best_placement is None:
                    # Can't place any brick
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
                    return {
                        "success": False,
                        "layers": None,
                        "error": f"Layer {z}: Can't fill {len(unfilled)} cells. Available: {inventory_status}",
                    }

                # Place the best brick
                x, y, brick_type, rotation = best_placement
                _place_brick(x, y, brick_type, rotation, layer_claimed, layer_bricks)
                inventory[brick_type] -= 1
                last_used_brick = brick_type

            all_layers.append(layer_bricks)

        # Check connectivity
        if not _check_connectivity(all_layers, voxel_grid.shape):
            return {
                "success": False,
                "layers": None,
                "error": "Structure has floating sections",
            }

        return {"success": True, "layers": all_layers, "error": None}


def _score_placement(
    x: int,
    y: int,
    brick_type: str,
    rotation: int,
    floating_cells: set,
    layer_below: list,
    inventory: dict,
    last_used_brick: str,
) -> float:
    """
    Score a placement based on multiple criteria.

    Returns:
        Score (higher is better)
    """
    score = 0.0

    # Get covered cells
    covered = _get_covered_cells(x, y, brick_type, rotation)

    # +5 for each floating cell covered
    floating_covered = len(covered & floating_cells)
    score += floating_covered * 5

    # +2 for not overlapping exactly with layer below
    if layer_below is not None:
        if not _overlaps_exactly_below(covered, layer_below):
            score += 2

    # +1 for each stud squared (heavily favors larger bricks)
    brick_area = BRICK_DIMS[brick_type][0] * BRICK_DIMS[brick_type][1]
    score += brick_area**2

    # +1 for each brick type with less remaining inventory
    inventory_rank = sum(
        1
        for other_type, other_count in inventory.items()
        if other_count < inventory[brick_type]
    )
    score += inventory_rank

    return score


def _overlaps_exactly_below(covered_cells: set, layer_below: list) -> bool:
    """
    Check if placement overlaps exactly with a brick in the layer below.
    """
    for brick in layer_below:
        brick_cells = _get_covered_cells(
            brick["x"], brick["y"], brick["type"], brick["rotation"]
        )

        # Check if covered_cells is exactly the same as brick_cells
        if covered_cells == brick_cells:
            return True

    return False


def _identify_floating_cells(voxel_grid: np.ndarray, z: int) -> set[tuple[int, int]]:
    """
    Identify floating cells in layer z.
    """
    width, depth, height = voxel_grid.shape
    floating = set()

    # Layer 0 is never floating (baseplate)
    if z == 0:
        return floating

    for x in range(width):
        for y in range(depth):
            if not voxel_grid[x, y, z]:
                continue

            has_neighbor_below = voxel_grid[x, y, z - 1]
            has_neighbor_above = z < height - 1 and voxel_grid[x, y, z + 1]

            if not has_neighbor_below and not has_neighbor_above:
                floating.add((x, y))

    return floating


def _find_all_valid_placements(
    brick_type: str,
    layer_voxels: np.ndarray,
    layer_claimed: np.ndarray,
    width: int,
    depth: int,
) -> list[tuple[int, int, int]]:
    """Find all valid placements for a brick type."""
    valid_placements = []

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
    """Check if a brick can be placed."""
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


def _get_covered_cells(
    x: int, y: int, brick_type: str, rotation: int
) -> set[tuple[int, int]]:
    """Get set of cells covered by a brick placement."""
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


def _place_brick(
    x: int,
    y: int,
    brick_type: str,
    rotation: int,
    layer_claimed: np.ndarray,
    layer_bricks: list,
) -> None:
    """Place a brick."""
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
    """Check if layer is complete."""
    width, depth = layer_voxels.shape

    for x in range(width):
        for y in range(depth):
            if layer_voxels[x, y] and not layer_claimed[x, y]:
                return False

    return True


def _check_connectivity(layers: list[list[dict]], grid_shape: tuple) -> bool:
    """Check connectivity."""
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
