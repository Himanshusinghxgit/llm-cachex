from __future__ import annotations

import os
import pickle
import logging
from typing import List, Dict

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS-based vector index for semantic search.
    Uses cosine similarity via normalized vectors.
    """

    def __init__(self, dim: int = 384, path: str = "faiss.index"):
        self.dim = dim
        self.path = path
        self.meta_path = f"{path}.meta"

        self.index = None
        self.id_map: List[str] = []

        self._load_or_create()

    # ---------------- INIT ----------------

    def _load_or_create(self):
        try:
            if os.path.exists(self.path) and os.path.exists(self.meta_path):
                logger.info("Loading FAISS index from disk")

                self.index = faiss.read_index(self.path)

                with open(self.meta_path, "rb") as f:
                    self.id_map = pickle.load(f)

            else:
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatIP(self.dim)
                self.id_map = []

        except Exception:
            logger.exception("Failed to initialize FAISS index")
            self.index = faiss.IndexFlatIP(self.dim)
            self.id_map = []

    # ---------------- ADD ----------------

    def add(self, vector: np.ndarray, doc_id: str):
        try:
            vector = self._prepare_vector(vector)

            self.index.add(vector)
            self.id_map.append(doc_id)

        except Exception:
            logger.exception("Failed to add vector to FAISS")

    def add_batch(self, vectors: np.ndarray, ids: List[str]):
        try:
            vectors = self._prepare_vector(vectors)

            self.index.add(vectors)
            self.id_map.extend(ids)

        except Exception:
            logger.exception("Batch insert failed")

    # ---------------- SEARCH ----------------

    def search(self, vector: np.ndarray, k: int = 3) -> List[Dict]:
        try:
            if self.index.ntotal == 0:
                return []

            vector = self._prepare_vector(vector)

            distances, indices = self.index.search(vector, k)

            results = []

            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue

                if idx >= len(self.id_map):
                    logger.warning(f"FAISS id_map mismatch: {idx}")
                    continue

                results.append({
                    "id": self.id_map[idx],
                    "score": float(distances[0][i])
                })

            return results

        except Exception:
            logger.exception("FAISS search failed")
            return []

    # ---------------- SAVE ----------------

    def save(self):
        """
        Persist index to disk (manual call recommended)
        """
        try:
            faiss.write_index(self.index, self.path)

            with open(self.meta_path, "wb") as f:
                pickle.dump(self.id_map, f)

            logger.info("FAISS index saved")

        except Exception:
            logger.exception("Failed to save FAISS index")

    # ---------------- UTIL ----------------

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = np.array(vector).astype("float32")

        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)

        faiss.normalize_L2(vector)
        return vector