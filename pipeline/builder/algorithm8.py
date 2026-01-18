"""
Algorithm 8: Connectivity-Aware Building
Builds layer by layer while ensuring each brick maintains structural connectivity.
"""

import random
from copy import deepcopy

import numpy as np

from .base import BRICK_DIMS, BRICKS_BY_SIZE, BuilderAlgorithm


class Algorithm8(BuilderAlgorithm):
    """
    Connectivity-aware building algorithm.

    Strategy:
    - Builds layer by layer from bottom to top
    - For each layer, only places bricks that connect to already-placed bricks
    - Layer 0 can place anywhere (rests on ground)
    - Layer N must overlap with at least one brick in layer N-1
    - This ensures connectivity by construction rather than validation
    """

    @property
    def name(self) -> str:
        return "Algorithm 8: Connectivity-Aware Building"

    @property
    def description(self) -> str:
        return (
            "Builds with connectivity awareness. "
            "Each brick must overlap with existing structure. "
            "Ensures all bricks connect to ground."
        )

    def build(self, voxels: np.ndarray, available_bricks: dict[str, int]) -> dict:
        """
        Build with connectivity awareness using multi-pass approach.

        Pass 1: Build from ground up (normal bottom-to-top connectivity)
        Pass 2: Build remaining cells that attach to existing structure (handles, overhangs)

        Tries multiple times with increasing tolerance for unfilled cells.
        """
        # Try with increasing tolerance
        for tolerance in range(8):
            for retry in range(3):
                result = self._attempt_build_multipass(
                    voxels, available_bricks.copy(), tolerance
                )
                if result["success"]:
                    result["total_tolerance"] = tolerance
                    return result

        return {
            "success": False,
            "error": "Failed to build with connectivity constraints",
            "layers": [],
        }

    def _attempt_build_multipass(
        self, voxels: np.ndarray, available_bricks: dict[str, int], tolerance: int
    ) -> dict:
        """
        Attempt multi-pass build.

        Pass 1: Ground-up (layer N must connect to layer N-1)
        Pass 2: Fill remaining by connecting to any existing brick (handles)
        """
        width, depth, height = voxels.shape
        total_voxels = voxels.sum()

        # Pass 1: Ground-up build
        print(f"      Pass 1 (ground-up): {total_voxels} voxels to fill")
        remaining_voxels = voxels.copy()
        result = self._attempt_build(
            remaining_voxels, available_bricks, tolerance, mode="ground-up"
        )

        if not result["success"]:
            print(f"      Pass 1 failed: {result.get('error', 'unknown')}")
            return result

        layers = result["layers"]
        total_cells_skipped = result.get("total_cells_skipped", 0)

        # Count how many voxels were filled in pass 1
        pass1_bricks = sum(len(layer) for layer in layers)
        print(f"      Pass 1 complete: {pass1_bricks} bricks placed")

        # Mark cells that were filled in pass 1
        for z, layer in enumerate(layers):
            for brick in layer:
                brick_w, brick_l = BRICK_DIMS[brick["type"]]
                if brick["rotation"] == 0:
                    dx, dy = brick_l, brick_w
                else:
                    dx, dy = brick_w, brick_l

                for i in range(dx):
                    for j in range(dy):
                        x_pos = brick["x"] + i
                        y_pos = brick["y"] + j
                        if (
                            0 <= x_pos < width
                            and 0 <= y_pos < depth
                            and 0 <= z < height
                        ):
                            remaining_voxels[x_pos, y_pos, z] = False

        # Check if anything remains
        remaining_count = remaining_voxels.sum()
        if not remaining_voxels.any():
            print(f"      Pass 1 filled everything!")
            return result

        # Pass 2: Fill remaining cells by attaching to existing structure
        print(f"      Pass 2 (attach): {remaining_count} voxels remaining")
        result2 = self._attempt_build(
            remaining_voxels,
            available_bricks,
            tolerance,
            mode="attach",
            existing_layers=layers,
        )

        if not result2["success"]:
            # Pass 2 failed, return pass 1 result (partial build within tolerance)
            print(f"      Pass 2 failed: {result2.get('error', 'unknown')}")
            if total_cells_skipped + remaining_count <= tolerance:
                print(
                    f"      Accepting partial build within tolerance ({total_cells_skipped + remaining_count}/{tolerance})"
                )
                result["total_cells_skipped"] = total_cells_skipped + remaining_count
                return result
            else:
                return result2  # Return the error

        # Merge pass 2 results into pass 1 layers
        pass2_bricks = sum(len(layer) for layer in result2["layers"])
        print(f"      Pass 2 complete: {pass2_bricks} bricks placed")

        for z, layer in enumerate(result2["layers"]):
            if z < len(layers):
                layers[z].extend(layer)
            else:
                layers.append(layer)

        total_bricks = sum(len(layer) for layer in layers)
        print(
            f"      Total: {total_bricks} bricks, {total_cells_skipped + result2.get('total_cells_skipped', 0)} cells skipped"
        )

        return {
            "success": True,
            "layers": layers,
            "error": None,
            "total_cells_skipped": total_cells_skipped
            + result2.get("total_cells_skipped", 0),
        }

    def _attempt_build(
        self,
        voxels: np.ndarray,
        available_bricks: dict[str, int],
        tolerance: int,
        mode: str = "ground-up",
        existing_layers: list = None,
    ) -> dict:
        """
        Attempt to build with connectivity awareness.

        Args:
            voxels: The voxel grid
            available_bricks: Dictionary of available bricks
            tolerance: Total cells allowed to skip across all layers
            mode: "ground-up" (connect to layer below) or "attach" (connect to any existing brick)
            existing_layers: For "attach" mode, the existing structure to attach to
        """
        width, depth, height = voxels.shape
        layers = []
        previous_layer_bricks = None
        total_cells_skipped = 0

        for z in range(height):
            layer_voxels = voxels[:, :, z].copy()
            layer_bricks = []

            if not layer_voxels.any():
                layers.append([])
                previous_layer_bricks = []
                continue

            # Build this layer with connectivity awareness
            while True:
                filled_count = layer_voxels.sum()

                if filled_count == 0:
                    break

                # Find best placement that maintains connectivity
                if mode == "ground-up":
                    placement = self._find_connected_placement(
                        layer_voxels, available_bricks, previous_layer_bricks, z
                    )
                else:  # "attach" mode
                    placement = self._find_attachment_placement(
                        layer_voxels, available_bricks, existing_layers, z
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
                            "error": f"Layer {z} cannot be completed with connectivity constraints",
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

    def _find_connected_placement(
        self,
        layer: np.ndarray,
        available_bricks: dict[str, int],
        previous_layer_bricks: list | None,
        layer_z: int,
    ) -> tuple | None:
        """
        Find best brick placement that maintains connectivity.

        For layer 0: Can place anywhere
        For layer N: Must overlap with at least one brick from layer N-1
        """
        best_score = -float("inf")
        best_placement = None

        width, depth = layer.shape

        # Try all brick types (largest first)
        brick_types = BRICKS_BY_SIZE

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
                        # Check if placement is valid (all cells available)
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

                        # Check connectivity requirement
                        if layer_z > 0 and previous_layer_bricks is not None:
                            # Must connect to previous layer
                            if not self._connects_to_previous_layer(
                                brick_type, x, y, rotation, previous_layer_bricks
                            ):
                                continue  # Skip this placement - doesn't connect

                        # Score this placement
                        score = self._score_placement(
                            brick_type, x, y, rotation, coverage, previous_layer_bricks
                        )

                        if score > best_score:
                            best_score = score
                            best_placement = (brick_type, x, y, rotation)

        return best_placement

    def _find_attachment_placement(
        self,
        layer: np.ndarray,
        available_bricks: dict[str, int],
        existing_layers: list,
        layer_z: int,
    ) -> tuple | None:
        """
        Find best brick placement that attaches to existing structure.

        Can connect to bricks in the same layer, layer above, or layer below.
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

                        # Check if it connects to existing structure
                        if not self._connects_to_existing_structure(
                            brick_type, x, y, rotation, existing_layers, layer_z
                        ):
                            continue

                        # Score this placement
                        score = self._score_placement(
                            brick_type, x, y, rotation, coverage, None
                        )

                        if score > best_score:
                            best_score = score
                            best_placement = (brick_type, x, y, rotation)

        return best_placement

    def _connects_to_existing_structure(
        self,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        existing_layers: list,
        layer_z: int,
    ) -> bool:
        """
        Check if a brick connects to the existing structure.

        Checks layer below, same layer, and layer above.
        """
        if not existing_layers:
            return False

        current_footprint = self._get_brick_footprint_from_params(
            brick_type, x, y, rotation
        )

        # Check layer below
        if layer_z > 0 and layer_z - 1 < len(existing_layers):
            for brick in existing_layers[layer_z - 1]:
                if current_footprint & self._get_brick_footprint(brick):
                    return True

        # Check same layer
        if layer_z < len(existing_layers):
            for brick in existing_layers[layer_z]:
                if current_footprint & self._get_brick_footprint(brick):
                    return True

        # Check layer above
        if layer_z + 1 < len(existing_layers):
            for brick in existing_layers[layer_z + 1]:
                if current_footprint & self._get_brick_footprint(brick):
                    return True

        return False

    def _connects_to_previous_layer(
        self,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        previous_layer_bricks: list,
    ) -> bool:
        """
        Check if a brick placement would connect to the previous layer.

        Returns True if the brick's footprint overlaps with any brick from previous layer.
        """
        if not previous_layer_bricks:
            return False

        current_footprint = self._get_brick_footprint_from_params(
            brick_type, x, y, rotation
        )

        for prev_brick in previous_layer_bricks:
            prev_footprint = self._get_brick_footprint(prev_brick)
            if current_footprint & prev_footprint:
                return True

        return False

    def _score_placement(
        self,
        brick_type: str,
        x: int,
        y: int,
        rotation: int,
        coverage: int,
        previous_layer_bricks: list | None,
    ) -> float:
        """
        Score a brick placement.

        Prioritizes:
        - Larger bricks
        - Full coverage
        - Not stacking same brick in same position
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
        brick_type = brick["type"]
        x = brick["x"]
        y = brick["y"]
        rotation = brick["rotation"]

        return self._get_brick_footprint_from_params(brick_type, x, y, rotation)

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
