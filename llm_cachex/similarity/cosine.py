from __future__ import annotations

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    NOTE:
    Not used in FAISS path (FAISS handles similarity internally).
    Useful for fallback / debugging / custom backends.
    """

    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denom)