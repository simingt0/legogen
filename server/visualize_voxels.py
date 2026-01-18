"""
Visualize voxel grids in 3D using matplotlib
Run from project root: python3 server/visualize_voxels.py <obj_path> [size]
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pipeline.voxelizer import voxelize_mesh


def visualize_voxels(voxels: np.ndarray, title: str = "Voxel Grid"):
    """
    Visualize a 3D voxel grid using matplotlib.

    Args:
        voxels: 3D boolean numpy array
        title: Title for the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Get coordinates of filled voxels
    filled = np.where(voxels)

    if len(filled[0]) == 0:
        print("Warning: No filled voxels to display")
        return

    # Create color array matching voxel grid shape
    # Color by height (z-axis) - bottom is dark, top is bright
    colors = np.zeros(voxels.shape + (4,))  # RGBA
    for x, y, z in zip(*filled):
        # Normalize z to 0-1 range
        z_norm = z / (voxels.shape[2] - 1) if voxels.shape[2] > 1 else 0.5
        # Get color from colormap
        color = plt.cm.viridis(z_norm)
        colors[x, y, z] = color

    # Plot filled voxels
    ax.voxels(voxels, facecolors=colors, edgecolors="gray", alpha=0.8, linewidth=0.5)

    # Set labels and title
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (height)")
    ax.set_title(title)

    # Set equal aspect ratio
    max_dim = max(voxels.shape)
    ax.set_xlim(0, max_dim)
    ax.set_ylim(0, max_dim)
    ax.set_zlim(0, max_dim)

    # Add info text
    filled_count = voxels.sum()
    total_count = voxels.size
    info = f"Shape: {voxels.shape}\n"
    info += f"Filled: {filled_count:,} / {total_count:,} ({100 * filled_count / total_count:.1f}%)"
    fig.text(
        0.02,
        0.98,
        info,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


def visualize_layers(voxels: np.ndarray, max_layers: int = None):
    """
    Visualize voxel grid layer by layer (2D slices).

    Args:
        voxels: 3D boolean numpy array
        max_layers: Maximum number of layers to show (None = all)
    """
    height = voxels.shape[2]

    if max_layers is None:
        max_layers = height

    num_layers = min(height, max_layers)

    # Create grid of subplots
    cols = 4
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Voxel Layers (Bottom to Top)", fontsize=16, y=0.995)

    for z in range(num_layers):
        row = z // cols
        col = z % cols
        ax = axes[row, col]

        layer = voxels[:, :, z]

        # Plot layer
        ax.imshow(layer.T, origin="lower", cmap="binary", interpolation="nearest")
        ax.set_title(f"Layer {z}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

        # Add count
        filled = layer.sum()
        total = layer.size
        ax.text(
            0.02,
            0.98,
            f"{filled}/{total}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Hide unused subplots
    for z in range(num_layers, rows * cols):
        row = z // cols
        col = z % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_combined(voxels: np.ndarray, title: str = "Voxel Grid"):
    """
    Show both 3D view and layer slices in one figure.
    """
    fig = plt.figure(figsize=(18, 8))

    # 3D view
    ax1 = fig.add_subplot(121, projection="3d")
    filled = np.where(voxels)
    if len(filled[0]) > 0:
        # Create color array matching voxel grid shape
        colors = np.zeros(voxels.shape + (4,))  # RGBA
        for x, y, z in zip(*filled):
            z_norm = z / (voxels.shape[2] - 1) if voxels.shape[2] > 1 else 0.5
            color = plt.cm.viridis(z_norm)
            colors[x, y, z] = color
        ax1.voxels(
            voxels, facecolors=colors, edgecolors="gray", alpha=0.8, linewidth=0.5
        )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D View")

    # Layer view (show a few key layers)
    height = voxels.shape[2]
    layer_indices = [0, height // 2, height - 1]
    layer_indices = [i for i in layer_indices if i < height]

    for idx, z in enumerate(layer_indices):
        ax = fig.add_subplot(3, 3, 4 + idx * 3)
        layer = voxels[:, :, z]
        ax.imshow(layer.T, origin="lower", cmap="binary", interpolation="nearest")
        ax.set_title(f"Layer {z}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 server/visualize_voxels.py <obj_path> [size] [mode]")
        print()
        print("Arguments:")
        print("  obj_path  - Path to OBJ file to voxelize")
        print("  size      - Voxel grid size (default: 16)")
        print(
            "  mode      - Visualization mode: '3d', 'layers', 'combined' (default: 3d)"
        )
        print()
        print("Examples:")
        print("  python3 server/visualize_voxels.py /tmp/legogen/model.obj")
        print("  python3 server/visualize_voxels.py /tmp/legogen/model.obj 24")
        print("  python3 server/visualize_voxels.py /tmp/legogen/model.obj 16 layers")
        sys.exit(1)

    obj_path = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    mode = sys.argv[3] if len(sys.argv) > 3 else "3d"

    if not Path(obj_path).exists():
        print(f"Error: File not found: {obj_path}")
        sys.exit(1)

    print(f"Voxelizing: {obj_path}")
    print(f"Size: {size}")
    print(f"Mode: {mode}")
    print("-" * 60)

    try:
        voxels = voxelize_mesh(obj_path, size=size)

        print(f"\n✅ Voxelization complete!")
        print(f"   Shape: {voxels.shape}")
        print(f"   Filled: {voxels.sum():,} / {voxels.size:,}")
        print("\nLaunching visualization...")

        title = f"{Path(obj_path).stem} (size={size})"

        if mode == "3d":
            visualize_voxels(voxels, title)
        elif mode == "layers":
            visualize_layers(voxels)
        elif mode == "combined":
            visualize_combined(voxels, title)
        else:
            print(f"Unknown mode: {mode}")
            print("Valid modes: 3d, layers, combined")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
