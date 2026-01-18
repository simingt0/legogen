"""
Algorithm 9: Floating Component Bonus
Incentivizes connecting disconnected components to the base through scoring bonuses.
"""

import random
from copy import deepcopy

import numpy as np

from .base import BRICK_DIMS, BRICKS_BY_SIZE, BuilderAlgorithm


class Algorithm9(BuilderAlgorithm):
    """
    Floating component bonus algorithm.

    Strategy:
    - Tracks connected and disconnected components during building
    - Base component: All bricks connected to layer 0 (ground)
    - Floating components: Groups of bricks not yet connected to base
    - Gives +4 per cell bonus for placements that connect floating components to base
    - This naturally encourages connectivity without enforcing it
    """

    @property
    def name(self) -> str:
        return "Algorithm 9: Floating Component Bonus"

    @property
    def description(self) -> str:
        return (
            "Incentivizes connectivity through scoring bonuses. "
            "Gives +4 per cell bonus for connecting floating components to base. "
            "Flexible approach that naturally builds connected structures."
        )

    def build(self, voxels: np.ndarray, available_bricks: dict[str, int]) -> dict:
        """
        Build with floating component bonus.

        Tries multiple times with increasing tolerance for unfilled cells.
        """
        # Try with increasing tolerance
        for tolerance in range(8):
            for retry in range(3):
                result = self._attempt_build(voxels, available_bricks.copy(), tolerance)
                if result["success"]:
                    result["total_tolerance"] = tolerance
                    return result

        return {
            "success": False,
            "error": "Failed to build after 24 attempts",
            "layers": [],
        }

    def _attempt_build(
        self, voxels: np.ndarray, available_bricks: dict[str, int], tolerance: int
    ) -> dict:
        """
        Attempt to build with floating component awareness.

        Args:
            voxels: The voxel grid
            available_bricks: Dictionary of available bricks
            tolerance: Total cells allowed to skip across all layers
        """
        width, depth, height = voxels.shape
        layers = []
        total_cells_skipped = 0

        for z in range(height):
            layer_voxels = voxels[:, :, z].copy()
            layer_bricks = []

            if not layer_voxels.any():
                layers.append([])
                continue

            # Build this layer
            while True:
                filled_count = layer_voxels.sum()

                if filled_count == 0:
                    break

                # Find best placement with floating component bonus
                placement = self._find_best_placement_with_bonus(
                    layer_voxels, available_bricks, layers, z
                )

                if placement is None:
                    # Can't place any brick
                    remaining = layer_voxels.sum()

                    if total_cells_skipped + remaining <= tolerance:
                        # Within tolerance - skip these cells
                        total_cells_skipped += remaining
                        break
                    else:
                        # Exceeds tolerance - fail
                        return {
                            "success": False,
                            "error": f"Layer {z} cannot be completed, exceeds tolerance",
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

        return {
            "success": True,
            "layers": layers,
            "error": None,
            "total_cells_skipped": total_cells_skipped,
        }

    def _find_best_placement_with_bonus(
        self,
        layer: np.ndarray,
        available_bricks: dict[str, int],
        existing_layers: list,
        layer_z: int,
    ) -> tuple | None:
        """
        Find best brick placement with floating component bonus.
        """
        best_score = -float("inf")
        best_placement = None

        width, depth = layer.shape
        brick_types = BRICKS_BY_SIZE

        for brick_type in brick_types:
            if available_bricks.get(brick_type, 0) <= 0:
                continue

            brick_w, brick_l = BRICK_DIMS[brick_type]

            for rotation in [0, 90]:
                if rotation == 0:
                    dx, dy = brick_l, brick_w
                else:
                    dx, dy = brick_w, brick_l

                for x in range(width - dx + 1):
                    for y in range(depth - dy + 1):
                        # Check if placement is valid
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
                            continue

                        # Score this placement with floating component bonus
                        score = self._score_placement_with_floating_bonus(
                            brick_type,
                            x,
                            y,
                            rotation,
                            coverage,
                            existing_layers,
                            layer_z,
                        )

                        if score > best_score:
                            best_score = score
                            best_placement = (brick_type, x, y, rotation)

        return best_placement

    def _score_placement_with_floating_bonus(
        self,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        coverage: int,
        existing_layers: list,
        layer_z: int,
    ) -> float:
        """
        Score a brick placement with floating component bonus.

        Base scoring + bonus if this connects floating components to base.
        """
        brick_w, brick_l = BRICK_DIMS[brick_type]
        brick_size = brick_w * brick_l

        # Base score: brick size
        score = brick_size * 10

        # Overlap efficiency bonus
        overlap_efficiency = coverage / brick_size
        score += overlap_efficiency * 50

        # Full coverage bonus
        if coverage == brick_size:
            score += 20

        # Slight penalty for small bricks
        if brick_size <= 2:
            score -= 5

        # Penalty for stacking same brick in same position
        if layer_z > 0 and layer_z - 1 < len(existing_layers):
            if self._is_exact_stack(
                brick_type, x, y, rotation, existing_layers[layer_z - 1]
            ):
                score -= 5

        # FLOATING COMPONENT BONUS
        # Check if this placement would connect floating components to base
        floating_bonus = self._calculate_floating_component_bonus(
            brick_type, x, y, rotation, existing_layers, layer_z
        )
        score += floating_bonus

        return score

    def _calculate_floating_component_bonus(
        self,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        existing_layers: list,
        layer_z: int,
    ) -> float:
        """
        Calculate bonus for connecting floating components to base.

        Returns +4 per cell for each floating brick that would be connected.
        """
        if not existing_layers or layer_z == 0:
            return 0.0  # Layer 0 is always connected

        # Get footprint of proposed brick
        proposed_footprint = self._get_brick_footprint_from_params(
            brick_type, x, y, rotation
        )

        # Identify which components this brick would connect to
        connected_to_base = False
        connected_floating_bricks = set()

        # Check layer below
        if layer_z > 0 and layer_z - 1 < len(existing_layers):
            for brick_idx, brick in enumerate(existing_layers[layer_z - 1]):
                brick_footprint = self._get_brick_footprint(brick)
                if proposed_footprint & brick_footprint:
                    # This brick connects to layer below
                    if self._is_brick_connected_to_base(
                        existing_layers, layer_z - 1, brick_idx
                    ):
                        connected_to_base = True
                    else:
                        connected_floating_bricks.add((layer_z - 1, brick_idx))

        # Check same layer (for current layer being built)
        if layer_z < len(existing_layers):
            for brick_idx, brick in enumerate(existing_layers[layer_z]):
                brick_footprint = self._get_brick_footprint(brick)
                if proposed_footprint & brick_footprint:
                    if self._is_brick_connected_to_base(
                        existing_layers, layer_z, brick_idx
                    ):
                        connected_to_base = True
                    else:
                        connected_floating_bricks.add((layer_z, brick_idx))

        # Check layer above
        if layer_z + 1 < len(existing_layers):
            for brick_idx, brick in enumerate(existing_layers[layer_z + 1]):
                brick_footprint = self._get_brick_footprint(brick)
                if proposed_footprint & brick_footprint:
                    if self._is_brick_connected_to_base(
                        existing_layers, layer_z + 1, brick_idx
                    ):
                        connected_to_base = True
                    else:
                        connected_floating_bricks.add((layer_z + 1, brick_idx))

        # If this brick connects to base AND floating components, give bonus
        if connected_to_base and connected_floating_bricks:
            # Calculate how many bricks in the floating component(s)
            floating_component_size = self._get_floating_component_size(
                existing_layers, connected_floating_bricks
            )
            # +4 per cell in the floating component that would be connected
            brick_w, brick_l = BRICK_DIMS[brick_type]
            brick_size = brick_w * brick_l
            bonus = floating_component_size * brick_size * 4
            return bonus

        return 0.0

    def _is_brick_connected_to_base(
        self, existing_layers: list, layer_z: int, brick_idx: int
    ) -> bool:
        """
        Check if a specific brick is connected to the base (layer 0).

        Uses flood fill to determine connectivity.
        """
        if layer_z == 0:
            return True  # Base layer is always connected

        # Simple connectivity check via flood fill
        visited = set()
        to_visit = [(layer_z, brick_idx)]
        visited.add((layer_z, brick_idx))

        while to_visit:
            z, idx = to_visit.pop()

            if z == 0:
                return True  # Reached base

            brick = existing_layers[z][idx]
            brick_footprint = self._get_brick_footprint(brick)

            # Check layer below
            if z > 0:
                for prev_idx, prev_brick in enumerate(existing_layers[z - 1]):
                    if (z - 1, prev_idx) in visited:
                        continue
                    prev_footprint = self._get_brick_footprint(prev_brick)
                    if brick_footprint & prev_footprint:
                        visited.add((z - 1, prev_idx))
                        to_visit.append((z - 1, prev_idx))

        return False

    def _get_floating_component_size(
        self, existing_layers: list, seed_bricks: set
    ) -> int:
        """
        Calculate size of floating component(s) starting from seed bricks.

        Returns total number of bricks in the floating component(s).
        """
        visited = set()
        to_visit = list(seed_bricks)
        component_size = 0

        while to_visit:
            z, idx = to_visit.pop()
            if (z, idx) in visited:
                continue

            visited.add((z, idx))
            component_size += 1

            # Don't count if connected to base
            if self._is_brick_connected_to_base(existing_layers, z, idx):
                continue

            brick = existing_layers[z][idx]
            brick_footprint = self._get_brick_footprint(brick)

            # Check adjacent bricks (above, below, same layer)
            for check_z in [z - 1, z, z + 1]:
                if 0 <= check_z < len(existing_layers):
                    for check_idx, check_brick in enumerate(existing_layers[check_z]):
                        if (check_z, check_idx) in visited:
                            continue
                        check_footprint = self._get_brick_footprint(check_brick)
                        if brick_footprint & check_footprint:
                            to_visit.append((check_z, check_idx))

        return component_size

    def _is_exact_stack(
        self,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        previous_layer_bricks: list,
    ) -> bool:
        """Check if exact same brick exists in previous layer."""
        for prev_brick in previous_layer_bricks:
            if (
                prev_brick["type"] == brick_type
                and prev_brick["x"] == x
                and prev_brick["y"] == y
                and prev_brick["rotation"] == rotation
            ):
                return True
        return False

    def _get_brick_footprint(self, brick: dict) -> set:
        """Get the set of (x, y) cells covered by a brick."""
        return self._get_brick_footprint_from_params(
            brick["type"], brick["x"], brick["y"], brick["rotation"]
        )

    def _get_brick_footprint_from_params(
        self, brick_type: str, x: int, y: int, rotation: int
    ) -> set:
        """Get the set of (x, y) cells covered by a brick from parameters."""
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
