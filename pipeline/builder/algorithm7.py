"""
Algorithm 7: Progressive Tolerance Building
Allows increasing tolerance for unfilled cells on retry attempts.
"""

import random
from copy import deepcopy

import numpy as np

from .base import BRICK_DIMS, BRICKS_BY_SIZE, BuilderAlgorithm


class Algorithm7(BuilderAlgorithm):
    """
    Progressive tolerance building algorithm.

    Strategy:
    - Attempt 1: Must fill all cells (0 total cells skipped)
    - Attempt 2: Can skip 1 cell total across ALL layers
    - Attempt 3: Can skip 2 cells total across ALL layers
    - Up to attempt 8: Can skip 7 cells total
    - Uses smart scoring with stacking penalties
    """

    @property
    def name(self) -> str:
        return "Algorithm 7: Progressive Tolerance Building"

    @property
    def description(self) -> str:
        return (
            "Progressive total tolerance building. "
            "Attempt N allows N-1 total cells skipped across all layers. "
            "Penalizes stacking same bricks in same positions."
        )

    def build(self, voxels: np.ndarray, available_bricks: dict[str, int]) -> dict:
        """
        Build with progressive total tolerance.

        Tries up to 8 times, allowing N-1 total skipped cells on attempt N.
        Also retries if connectivity validation fails.
        """
        # Try with increasing total tolerance
        last_error = None
        connectivity_failures = 0
        brick_failures = 0

        for attempt in range(1, 9):  # Attempts 1-8
            total_tolerance = attempt - 1  # 0, 1, 2, ..., 7

            # Try multiple times per tolerance level (for randomization/connectivity issues)
            for retry in range(3):
                result = self._attempt_build(
                    voxels, available_bricks.copy(), total_tolerance
                )

                if result["success"]:
                    # Validate structural connectivity
                    if self._validate_connectivity(result["layers"], voxels.shape):
                        result["total_cells_skipped"] = result.get(
                            "total_cells_skipped", 0
                        )
                        result["total_tolerance"] = total_tolerance
                        print(
                            f"   Build succeeded with tolerance={total_tolerance}, attempt={attempt}, retry={retry}"
                        )
                        return result
                    else:
                        connectivity_failures += 1
                        last_error = "connectivity"
                        print(
                            f"   [Tolerance {total_tolerance}, retry {retry}] Failed connectivity check"
                        )
                else:
                    brick_failures += 1
                    last_error = result.get("error", "insufficient bricks")
                    print(
                        f"   [Tolerance {total_tolerance}, retry {retry}] Insufficient bricks: {result.get('error', '')[:80]}"
                    )

        # All attempts failed
        print(
            f"\n   Total attempts: {brick_failures} brick failures, {connectivity_failures} connectivity failures"
        )
        return {
            "success": False,
            "error": f"Failed after 24 attempts - {brick_failures} brick failures, {connectivity_failures} connectivity failures",
            "layers": [],
        }

    def _attempt_build(
        self, voxels: np.ndarray, available_bricks: dict[str, int], total_tolerance: int
    ) -> dict:
        """
        Attempt to build with a specific total tolerance.

        Args:
            voxels: The voxel grid
            available_bricks: Dictionary of available bricks
            total_tolerance: Total number of cells allowed to be skipped across ALL layers
        """
        width, depth, height = voxels.shape
        layers = []
        previous_layer_bricks = None
        total_cells_skipped = 0  # Track total skipped across all layers

        for z in range(height):
            layer_voxels = voxels[:, :, z].copy()
            layer_bricks = []

            if not layer_voxels.any():
                layers.append([])
                continue

            # Build this layer
            while True:
                filled_count = layer_voxels.sum()

                # Check if layer is complete
                if filled_count == 0:
                    break

                # Try to place a brick
                placement = self._find_best_placement(
                    layer_voxels,
                    available_bricks,
                    previous_layer_bricks,
                )

                if placement is None:
                    # Can't place any brick
                    remaining = layer_voxels.sum()

                    # Check if we can skip these cells within total tolerance
                    if total_cells_skipped + remaining <= total_tolerance:
                        # Within tolerance - skip these cells and continue
                        total_cells_skipped += remaining
                        break
                    else:
                        # Exceeds total tolerance - fail
                        return {
                            "success": False,
                            "error": f"Layer {z} has {remaining} unfilled cells, total skipped would be {total_cells_skipped + remaining}, exceeds tolerance of {total_tolerance}",
                            "layers": [],
                        }

                # Place the brick
                brick_type, x, y, rotation = placement
                layer_bricks.append(
                    {
                        "type": brick_type,
                        "x": int(x),
                        "y": int(y),
                        "rotation": int(rotation),
                    }
                )

                # Mark cells as filled
                brick_w, brick_l = BRICK_DIMS[brick_type]
                if rotation == 0:
                    dx, dy = brick_l, brick_w
                else:
                    dx, dy = brick_w, brick_l

                for i in range(dx):
                    for j in range(dy):
                        if 0 <= x + i < width and 0 <= y + j < depth:
                            layer_voxels[x + i, y + j] = False

                # Decrement brick count
                available_bricks[brick_type] -= 1

            layers.append(layer_bricks)
            previous_layer_bricks = layer_bricks

        return {
            "success": True,
            "layers": layers,
            "error": None,
            "total_cells_skipped": total_cells_skipped,
        }

    def _find_best_placement(
        self,
        layer: np.ndarray,
        available_bricks: dict[str, int],
        previous_layer_bricks: list | None,
    ) -> tuple | None:
        """
        Find the best brick placement for the current layer.

        Uses a scoring system that prioritizes:
        1. Larger bricks
        2. Bricks that cover floating cells
        3. Bricks with good overlap
        """
        best_score = -float("inf")
        best_placement = None

        width, depth = layer.shape

        # Try all brick types in order of preference (largest first)
        brick_types = BRICKS_BY_SIZE  # Already sorted by size descending

        for brick_type in brick_types:
            if available_bricks.get(brick_type, 0) <= 0:
                continue

            brick_w, brick_l = BRICK_DIMS[brick_type]

            # Try both rotations
            for rotation in [0, 90]:
                if rotation == 0:
                    dx, dy = brick_l, brick_w
                else:
                    dx, dy = brick_w, brick_l

                # Try all positions
                for x in range(width - dx + 1):
                    for y in range(depth - dy + 1):
                        # Check if placement is valid - ALL cells must be available (True)
                        valid = True
                        coverage = 0
                        for i in range(dx):
                            for j in range(dy):
                                if not layer[x + i, y + j]:
                                    valid = False
                                    break
                                coverage += 1
                            if not valid:
                                break

                        if not valid or coverage == 0:
                            continue  # Placement overlaps or doesn't cover any voxels

                        # Score this placement
                        score = self._score_placement(
                            layer,
                            brick_type,
                            x,
                            y,
                            rotation,
                            available_bricks,
                            previous_layer_bricks,
                        )

                        if score > best_score:
                            best_score = score
                            best_placement = (brick_type, x, y, rotation)

        return best_placement

    def _score_placement(
        self,
        layer: np.ndarray,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        available_bricks: dict[str, int],
        previous_layer_bricks: list | None,
    ) -> float:
        """
        Score a brick placement.

        Scoring factors:
        - Brick size (studs²)
        - Coverage (how many voxels it covers)
        - Overlap efficiency (coverage / brick_size)
        """
        brick_w, brick_l = BRICK_DIMS[brick_type]
        brick_size = brick_w * brick_l

        if rotation == 0:
            dx, dy = brick_l, brick_w
        else:
            dx, dy = brick_w, brick_l

        # Check if all cells are available and count coverage
        coverage = 0
        for i in range(dx):
            for j in range(dy):
                if not layer[x + i, y + j]:
                    # Cell already occupied
                    return -float("inf")
                coverage += 1

        if coverage == 0:
            return -float("inf")

        # Calculate overlap efficiency
        overlap_efficiency = coverage / brick_size

        # Base score: brick size
        score = brick_size * 10

        # Bonus for high overlap efficiency
        score += overlap_efficiency * 50

        # Bonus for full coverage
        if coverage == brick_size:
            score += 20

        # Slight penalty for small bricks to preserve them
        if brick_size <= 2:
            score -= 5

        # Penalty for stacking exact same brick in exact same position
        if previous_layer_bricks is not None:
            if self._is_exact_stack(brick_type, x, y, rotation, previous_layer_bricks):
                score -= 5

        return score

    def _is_exact_stack(
        self,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        previous_layer_bricks: list,
    ) -> bool:
        """
        Check if this brick is the exact same type in the exact same position
        as a brick in the previous layer.

        Returns True if exact match found, False otherwise.
        """
        for prev_brick in previous_layer_bricks:
            if (
                prev_brick["type"] == brick_type
                and prev_brick["x"] == x
                and prev_brick["y"] == y
                and prev_brick["rotation"] == rotation
            ):
                return True

        return False

    def _validate_connectivity(self, layers: list, voxel_shape: tuple) -> bool:
        """
        Validate that all bricks are structurally connected.

        Each brick must satisfy at least ONE of:
        - Be in layer 0 (rests on ground)
        - Connect to a brick above it that's part of the main component
        - Connect to a brick below it that's part of the main component

        Uses bidirectional flood-fill to find the main connected component.

        Args:
            layers: List of layer brick placements
            voxel_shape: Shape of voxel grid (width, depth, height)

        Returns:
            True if all bricks are part of main component, False otherwise
        """
        if not layers or all(len(layer) == 0 for layer in layers):
            return True  # No bricks to validate

        # Build brick lookup
        all_bricks = []
        for z, layer in enumerate(layers):
            for brick_idx, brick in enumerate(layer):
                all_bricks.append((z, brick_idx, brick))

        if not all_bricks:
            return True

        # Precompute footprints for all bricks
        footprints = {}
        for z, brick_idx, brick in all_bricks:
            footprints[(z, brick_idx)] = self._get_brick_footprint(brick)

        # Start from layer 0 (ground) and flood fill both up AND down
        connected = set()
        to_visit = []

        # Add all layer 0 bricks to seed
        for z, brick_idx, brick in all_bricks:
            if z == 0:
                connected.add((z, brick_idx))
                to_visit.append((z, brick_idx))

        # Bidirectional flood fill
        while to_visit:
            curr_z, curr_idx = to_visit.pop()
            curr_footprint = footprints[(curr_z, curr_idx)]

            # Check bricks in layer ABOVE
            if curr_z + 1 < len(layers):
                for next_idx in range(len(layers[curr_z + 1])):
                    if (curr_z + 1, next_idx) in connected:
                        continue

                    next_footprint = footprints[(curr_z + 1, next_idx)]
                    if curr_footprint & next_footprint:
                        connected.add((curr_z + 1, next_idx))
                        to_visit.append((curr_z + 1, next_idx))

            # Check bricks in layer BELOW
            if curr_z - 1 >= 0:
                for prev_idx in range(len(layers[curr_z - 1])):
                    if (curr_z - 1, prev_idx) in connected:
                        continue

                    prev_footprint = footprints[(curr_z - 1, prev_idx)]
                    if curr_footprint & prev_footprint:
                        connected.add((curr_z - 1, prev_idx))
                        to_visit.append((curr_z - 1, prev_idx))

        # All bricks must be in the connected component
        return len(connected) == len(all_bricks)

    def _get_brick_footprint(self, brick: dict) -> set:
        """
        Get the set of (x, y) cells covered by a brick.

        Args:
            brick: Brick placement dict with type, x, y, rotation

        Returns:
            Set of (x, y) tuples representing cells covered
        """
        brick_type = brick["type"]
        x = brick["x"]
        y = brick["y"]
        rotation = brick["rotation"]

        brick_w, brick_l = BRICK_DIMS[brick_type]

        if rotation == 0:
            dx, dy = brick_l, brick_w
        else:
            dx, dy = brick_w, brick_l

        cells = set()
        for i in range(dx):
            for j in range(dy):
                cells.add((x + i, y + j))

        return cells
