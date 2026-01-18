"""
Base algorithm interface for LEGO builder algorithms
All algorithm implementations must follow this interface
"""

from abc import ABC, abstractmethod

import numpy as np


class BuilderAlgorithm(ABC):
    """
    Abstract base class for LEGO building algorithms.

    All algorithms must implement the build() method which takes
    a voxel grid and available bricks and returns build instructions.
    """

    @abstractmethod
    def build(
        self,
        voxel_grid: np.ndarray,
        available_bricks: dict[str, int],
    ) -> dict:
        """
        Generate LEGO build instructions from a voxel grid.

        Args:
            voxel_grid: 3D boolean numpy array (width, depth, height)
                       True = filled voxel, False = empty
            available_bricks: Dict mapping brick types to counts
                            e.g., {"2x4": 12, "1x1": 30}
                            This dict will be modified (decremented) during building

        Returns:
            {
                "success": bool,
                "layers": [                    # List of layers, bottom to top
                    [                          # Each layer is a list of brick placements
                        {
                            "type": "2x4",     # Brick type
                            "x": 0,            # X position (left edge)
                            "y": 0,            # Y position (front edge)
                            "rotation": 0      # 0 = length along X, 90 = length along Y
                        },
                        ...
                    ],
                    ...
                ],
                "error": str | None           # Error message if success=False
            }
        """
        pass

    @property
    def name(self) -> str:
        """Return the name/description of this algorithm"""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Return a description of how this algorithm works"""
        return "No description provided"


# Brick dimensions: (width, length) where width <= length
BRICK_DIMS = {
    "1x1": (1, 1),
    "1x2": (1, 2),
    "1x3": (1, 3),
    "1x4": (1, 4),
    "1x6": (1, 6),
    "2x2": (2, 2),
    "2x3": (2, 3),
    "2x4": (2, 4),
    "2x6": (2, 6),
}

# Sorted by area descending for placement priority
BRICKS_BY_SIZE = sorted(
    BRICK_DIMS.keys(), key=lambda b: BRICK_DIMS[b][0] * BRICK_DIMS[b][1], reverse=True
)
