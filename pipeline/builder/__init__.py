"""
LEGO Builder Module - Multiple algorithm implementations
"""

from .algorithm0 import Algorithm0, generate_build_instructions
from .algorithm1 import Algorithm1
from .algorithm2 import Algorithm2
from .algorithm3 import Algorithm3
from .algorithm4 import Algorithm4
from .algorithm5 import Algorithm5
from .algorithm6 import Algorithm6
from .base import BRICK_DIMS, BRICKS_BY_SIZE, BuilderAlgorithm

# Available algorithms
ALGORITHMS = {
    "algorithm0": Algorithm0,
    "algorithm1": Algorithm1,
    "algorithm2": Algorithm2,
    "algorithm3": Algorithm3,
    "algorithm4": Algorithm4,
    "algorithm5": Algorithm5,
    "algorithm6": Algorithm6,
}


def get_algorithm(name: str = "algorithm0") -> BuilderAlgorithm:
    """
    Get an algorithm instance by name.

    Args:
        name: Algorithm name (default: "algorithm0")

    Returns:
        BuilderAlgorithm instance

    Raises:
        ValueError: If algorithm name is not recognized
    """
    if name not in ALGORITHMS:
        available = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm: {name}. Available: {available}")

    return ALGORITHMS[name]()


def list_algorithms() -> dict[str, str]:
    """
    List all available algorithms with their descriptions.

    Returns:
        Dict mapping algorithm names to descriptions
    """
    result = {}
    for name, algo_class in ALGORITHMS.items():
        instance = algo_class()
        result[name] = f"{instance.name}: {instance.description}"
    return result


__all__ = [
    "BuilderAlgorithm",
    "BRICK_DIMS",
    "BRICKS_BY_SIZE",
    "generate_build_instructions",
    "get_algorithm",
    "list_algorithms",
    "ALGORITHMS",
    "Algorithm0",
    "Algorithm1",
    "Algorithm2",
    "Algorithm3",
    "Algorithm4",
    "Algorithm5",
    "Algorithm6",
]
