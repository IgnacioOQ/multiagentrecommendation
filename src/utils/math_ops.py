"""
Mathematical operations for RL agents.

This module provides efficient matrix operations used by various agents.
"""

import numpy as np


def sherman_morrison_update(A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Update matrix inverse using Sherman-Morrison formula.

    Given A^{-1} and a vector x, computes (A + x x^T)^{-1} in O(d²) time
    instead of O(d³) for direct inversion.

    Sherman-Morrison formula:
        (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)

    Args:
        A_inv: Current matrix inverse of shape (d, d).
        x: Update vector of shape (d,).

    Returns:
        Updated matrix inverse of shape (d, d).
    """
    x = x.flatten()

    # Compute A^{-1} x
    A_inv_x = A_inv @ x

    # Compute denominator: 1 + x^T A^{-1} x
    denom = 1.0 + np.dot(x, A_inv_x)

    # Update: A^{-1} - (A^{-1} x)(A^{-1} x)^T / denom
    A_inv_new = A_inv - np.outer(A_inv_x, A_inv_x) / denom

    return A_inv_new
