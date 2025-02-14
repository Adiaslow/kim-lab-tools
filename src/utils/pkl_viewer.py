"""
Utility functions for viewing and analyzing ROI pickle files.
"""

import pickle
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any


def create_sparse_matrix(
    roi_dict: Dict[Tuple[int, int], int]
) -> Tuple[sparse.csr_matrix, Tuple[int, int]]:
    """
    Create a sparse matrix from ROI coordinate dictionary.

    Parameters:
    -----------
    roi_dict : dict
        Dictionary with (y, x) coordinate tuples as keys and intensity values as values

    Returns:
    --------
    sparse.csr_matrix
        Sparse matrix representation of the ROI
    tuple
        (y_min, x_min) offset values for coordinate adjustment
    """
    # Extract coordinates and values
    coords = np.array(list(roi_dict.keys()))
    values = np.array(list(roi_dict.values()), dtype=np.uint8)

    # Find image boundaries
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    # Adjust coordinates to start from 0
    rows = coords[:, 0] - y_min
    cols = coords[:, 1] - x_min

    # Create sparse matrix
    shape = (y_max - y_min, x_max - x_min)
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=shape)

    return matrix, (y_min, x_min)


def load_and_display_pkl(file_path: str) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """
    Load a pickle file containing ROI data and display it efficiently.

    Parameters:
    -----------
    file_path : str or Path
        Path to the pickle file

    Returns:
    --------
    tuple
        (sparse_matrix, metadata)
        - sparse_matrix: scipy.sparse.csr_matrix containing the ROI data
        - metadata: dictionary containing file metadata
    """
    # Load the pickle file
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Create sparse matrix from ROI data
    roi_matrix, (y_min, x_min) = create_sparse_matrix(data["roi"])

    # Get matrix properties
    matrix_shape = roi_matrix.shape
    n_nonzero = roi_matrix.nnz  # Number of non-zero elements
    sparsity = 1.0 - (n_nonzero / (matrix_shape[0] * matrix_shape[1]))

    # Display the image
    plt.figure(figsize=(12, 8))

    # Create subplot layout
    plt.subplot(121)
    plt.imshow(roi_matrix.toarray(), cmap="gray")
    plt.colorbar(label="Intensity")
    plt.title(f"ROI: {data['name']}" if "name" in data else "ROI Image")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    # Add histogram of non-zero values
    plt.subplot(122)
    nonzero_vals = roi_matrix.data
    plt.hist(nonzero_vals, bins=50, color="blue", alpha=0.7)
    plt.title("Intensity Distribution")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")

    # Add text with statistics
    stats_text = (
        f"Image Shape: {matrix_shape}\n"
        f"Non-zero pixels: {n_nonzero:,}\n"
        f"Sparsity: {sparsity:.2%}\n"
        f"Value range: [{nonzero_vals.min()}, {nonzero_vals.max()}]"
    )
    plt.text(
        1.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.show()

    return roi_matrix, data


def analyze_roi_structure(file_path: str) -> None:
    """
    Analyze and print the structure of a ROI pickle file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the pickle file
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"\nStructure of pickle file: {file_path}\n")
    print(f"Type: {type(data).__name__}")
    print(f"Shape/Length: {len(data)}")
    print(f"Keys: {list(data.keys())}")

    roi_dict = data["roi"]
    coords = np.array(list(roi_dict.keys()))
    values = np.array(list(roi_dict.values()))

    print("\nROI Statistics:")
    print(f"Total coordinates: {len(coords):,}")
    print(
        f"Coordinate range: y[{coords[:, 0].min()}, {coords[:, 0].max()}], "
        f"x[{coords[:, 1].min()}, {coords[:, 1].max()}]"
    )
    print(f"Value range: [{values.min()}, {values.max()}]")
    print(f"Mean intensity: {values.mean():.2f}")
    print(f"Median intensity: {np.median(values):.2f}")


def main():
    # Example usage
    file_path = "/Volumes/euiseokdataUCSC_1/Matt_Jacobs/Images_and_Data/H2B_quantification/p60/m776/M776_s030_RSPagl.pkl"

    try:
        # Analyze file structure
        analyze_roi_structure(file_path)

        # Load and display the ROI
        roi_matrix, metadata = load_and_display_pkl(file_path)

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
