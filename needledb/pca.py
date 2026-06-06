import numpy as np
from typing import Tuple


def pca_2d(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project N-dimensional vectors to 2D using PCA from scratch.

    ── WHAT IS PCA? ─────────────────────────────────────────────────────
    Principal Component Analysis finds the directions of maximum
    variance in high-dimensional data and projects onto them.
    We keep the top 2 → a 2D scatter plot humans can actually see.

    If semantic clustering is strong, similar vectors (all CS topics,
    all food topics) will visibly group together on the plot.
    This is the visual proof that the embedding space works.

    ── STEPS ────────────────────────────────────────────────────────────
    1. Center  : subtract the mean vector so data is centered at origin
    2. Covariance : compute d×d covariance matrix (how dims vary together)
    3. Eigen   : decompose covariance → eigenvalues + eigenvectors
    4. Select  : pick the 2 eigenvectors with the LARGEST eigenvalues
                 (they capture the most variance = most information)
    5. Project : dot product of centered data with those 2 eigenvectors

    Args:
        vectors : np.ndarray of shape (N, d)

    Returns:
        projected        : np.ndarray of shape (N, 2)  — 2D coordinates
        explained_ratio  : np.ndarray of shape (2,)    — variance explained
                           by each principal component (sums to ≤ 1.0)
    """
    n = len(vectors)

    if n == 0:
        return np.zeros((0, 2)), np.array([0.0, 0.0])
    if n == 1:
        return np.zeros((1, 2)), np.array([0.0, 0.0])

    vectors = np.array(vectors, dtype=np.float64)

    # Step 1: Center the data
    mean = np.mean(vectors, axis=0)
    X    = vectors - mean

    # Step 2: Covariance matrix  (d × d)
    cov = np.cov(X.T)

    # Edge case: single dimension
    if cov.ndim == 0:
        return np.zeros((n, 2)), np.array([0.0, 0.0])

    # Step 3: Eigendecomposition
    # np.linalg.eigh is stable for symmetric matrices (covariance is always symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4: Sort descending, keep top 2 principal components
    idx        = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:2]]       # shape: (d, 2)

    # Explained variance ratio
    total_var = np.sum(np.abs(eigenvalues))
    explained = eigenvalues[idx[:2]] / total_var if total_var > 0 else np.array([0.0, 0.0])

    # Step 5: Project onto the 2 principal components
    projected = X @ components                  # shape: (N, 2)

    return projected, np.abs(explained)
