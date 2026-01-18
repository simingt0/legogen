"""
Algorithm 6: Beam Search Layer Optimization
Generates multiple complete layer candidates and selects the highest scoring one.
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


class Algorithm6(BuilderAlgorithm):
    """
    Beam search layer optimization algorithm.

    Strategy:
    - For each layer, generate K complete candidate layouts (beam width)
    - Each candidate built by selecting from top-N scoring placements
    - Score entire layer (sum of all brick placement scores)
    - Keep the best scoring complete layer
    - Uses same scoring as Algorithm 5 but optimizes at layer level
    """

    @property
    def name(self) -> str:
        return "Algorithm 6: Beam Search Layer Optimization"

    @property
    def description(self) -> str:
        return (
            "Generates multiple complete layer candidates using beam search. "
            "Selects top-N scoring placements at each step, scores entire layer, "
            "keeps best. Same brick scoring as Algorithm 5."
        )

    def build(
        self,
        voxel_grid: np.ndarray,
        available_bricks: dict[str, int],
    ) -> dict:
        """Generate LEGO build instructions using beam search layer optimization."""

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

        # Beam search parameters
        beam_width = 3  # Number of candidates to try per layer
        top_n = 2  # Select from top N placements at each step

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

            # Generate multiple candidate layers
            best_layer = None
            best_layer_score = -float("inf")

            for candidate_idx in range(beam_width):
                # Try to build a complete candidate layer
                candidate_result = _build_candidate_layer(
                    layer_voxels,
                    floating_cells,
                    layer_below,
                    inventory.copy(),
                    width,
                    depth,
                    top_n,
                )

                if candidate_result["success"]:
                    # Score this complete layer
                    layer_score = candidate_result["score"]

                    if layer_score > best_layer_score:
                        best_layer_score = layer_score
                        best_layer = candidate_result

            if best_layer is None:
                # All candidates failed
                inventory_status = ", ".join(
                    f"{bt}: {cnt}" for bt, cnt in sorted(inventory.items()) if cnt > 0
                )
                return {
                    "success": False,
                    "layers": None,
                    "error": f"Layer {z}: All {beam_width} candidates failed. Available: {inventory_status}",
                }

            # Use the best layer and update global inventory
            all_layers.append(best_layer["bricks"])

            # Update inventory based on what was used
            for brick in best_layer["bricks"]:
                inventory[brick["type"]] -= 1

        # Check connectivity
        if not _check_connectivity(all_layers, voxel_grid.shape):
            return {
                "success": False,
                "layers": None,
                "error": "Structure has floating sections",
            }

        return {"success": True, "layers": all_layers, "error": None}


def _build_candidate_layer(
    layer_voxels: np.ndarray,
    floating_cells: set,
    layer_below: list,
    inventory: dict,
    width: int,
    depth: int,
    top_n: int,
) -> dict:
    """
    Build a single candidate layer using greedy randomized selection.

    Returns:
        {
            "success": bool,
            "bricks": list of brick placements,
            "score": total layer score
        }
    """
    layer_bricks = []
    layer_claimed = np.zeros((width, depth), dtype=bool)
    total_score = 0.0

    # Build layer brick by brick
    while not _is_layer_complete(layer_voxels, layer_claimed):
        # Collect all possible placements with scores
        all_placements = []

        for brick_type in CUSTOM_BRICK_ORDER:
            if inventory.get(brick_type, 0) <= 0:
                continue

            # Find all valid placements for this brick
            valid_placements = _find_all_valid_placements(
                brick_type, layer_voxels, layer_claimed, width, depth
            )

            for x, y, rotation in valid_placements:
                score = _score_placement(
                    x, y, brick_type, rotation, floating_cells, layer_below, inventory
                )

                all_placements.append((score, x, y, brick_type, rotation))

        if not all_placements:
            # Can't place any brick - candidate failed
            return {"success": False, "bricks": [], "score": 0}

        # Sort by score and pick from top N
        all_placements.sort(reverse=True, key=lambda p: p[0])
        top_placements = all_placements[: min(top_n, len(all_placements))]

        # Randomly select from top N
        selected = random.choice(top_placements)
        score, x, y, brick_type, rotation = selected

        # Place the brick
        _place_brick(x, y, brick_type, rotation, layer_claimed, layer_bricks)
        inventory[brick_type] -= 1
        total_score += score

    return {"success": True, "bricks": layer_bricks, "score": total_score}


def _score_placement(
    x: int,
    y: int,
    brick_type: str,
    rotation: int,
    floating_cells: set,
    layer_below: list,
    inventory: dict,
) -> float:
    """Score a placement (same as Algorithm 5)."""
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

    # -10 penalty for 1x1 and 1x2 to preserve them as gap fillers
    if brick_type in ["1x1", "1x2"]:
        score -= 10

    # +1 for each brick type with less remaining inventory
    inventory_rank = sum(
        1
        for other_type, other_count in inventory.items()
        if other_count < inventory[brick_type]
    )
    score += inventory_rank

    return score


def _overlaps_exactly_below(covered_cells: set, layer_below: list) -> bool:
    """Check if placement overlaps exactly with a brick in the layer below."""
    for brick in layer_below:
        brick_cells = _get_covered_cells(
            brick["x"], brick["y"], brick["type"], brick["rotation"]
        )

        if covered_cells == brick_cells:
            return True

    return False


def _identify_floating_cells(voxel_grid: np.ndarray, z: int) -> set[tuple[int, int]]:
    """Identify floating cells in layer z."""
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
