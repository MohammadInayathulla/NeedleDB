import numpy as np

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance = 1 - cosine_similarity.
    Range: [0, 2]. Lower = more similar.
    Best for semantic similarity (text embeddings).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Straight-line distance between two points in N-dimensional space.
    Range: [0, inf]. Lower = more similar.
    Best for image embeddings, spatial data.
    """
    return float(np.linalg.norm(a - b))

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Sum of absolute differences across all dimensions.
    Range: [0, inf]. Lower = more similar.
    More robust to outliers than euclidean.
    """
    return float(np.sum(np.abs(a - b)))


METRICS = {
    "cosine":     cosine_distance,
    "euclidean":  euclidean_distance,
    "manhattan":  manhattan_distance,
}

def get_metric(name: str):
    if name not in METRICS:
        raise ValueError(f"Unknown metric '{name}'. Choose from: {list(METRICS.keys())}")
    return METRICS[name]
