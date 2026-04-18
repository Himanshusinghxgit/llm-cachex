from __future__ import annotations

import logging
from typing import List, Union, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedding wrapper with lazy loading and batch support.
    Supports future extension (OpenAI embeddings, etc.)
    """

    _model_instance: Optional[SentenceTransformer] = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device

        if Embedder._model_instance is None:
            self._load_model()

        self.model = Embedder._model_instance

    # ---------------- INIT ----------------

    def _load_model(self):
        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            Embedder._model_instance = SentenceTransformer(
                self.model_name,
                device=self.device
            )

        except Exception:
            logger.exception("Failed to load embedding model")
            raise

    # ---------------- ENCODE ----------------

    def encode(
        self,
        text: Union[str, List[str]],
        normalize: bool = False
    ):
        """
        Encode single string or list of strings.
        """

        try:
            embeddings = self.model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            return embeddings

        except Exception:
            logger.exception("Embedding failed")
            raise