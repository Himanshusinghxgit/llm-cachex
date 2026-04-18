from __future__ import annotations

import os
from typing import Optional


class CacheConfig:
    """
    Central configuration for LLM cache behavior.
    """

    def __init__(
        self,
        similarity_threshold: float = None,
        enable_semantic: bool = None,
        enable_exact: bool = None,
        enable_metrics: bool = None,
        enable_token_cost: bool = None,
        model_name: Optional[str] = None,
        top_k: int = None,
        # ttl: int = None,
        ttl: int =3600,
    ):
        # -------- ENV DEFAULTS --------
        self.similarity_threshold = float(
            similarity_threshold if similarity_threshold is not None
            else os.getenv("LLM_CACHE_SIM_THRESHOLD", 0.7)
        )

        self.enable_semantic = bool(
            enable_semantic if enable_semantic is not None
            else os.getenv("LLM_CACHE_ENABLE_SEMANTIC", "true").lower() == "true"
        )

        self.enable_exact = bool(
            enable_exact if enable_exact is not None
            else os.getenv("LLM_CACHE_ENABLE_EXACT", "true").lower() == "true"
        )

        self.enable_metrics = bool(
            enable_metrics if enable_metrics is not None
            else os.getenv("LLM_CACHE_ENABLE_METRICS", "true").lower() == "true"
        )

        self.enable_token_cost = bool(
            enable_token_cost if enable_token_cost is not None
            else os.getenv("LLM_CACHE_ENABLE_TOKEN_COST", "true").lower() == "true"
        )

        self.model_name = model_name or os.getenv("LLM_CACHE_MODEL")

        self.top_k = int(
            top_k if top_k is not None
            else os.getenv("LLM_CACHE_TOP_K", 3)
        )

        self.ttl = int(
            ttl if ttl is not None
            else os.getenv("LLM_CACHE_TTL", 3600)
        )

        # -------- VALIDATION --------
        self._validate()

    # ---------------- VALIDATION ----------------

    def _validate(self):
        if not (0 <= self.similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be between 0 and 1")

        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")

        if self.ttl <= 0:
            raise ValueError("ttl must be > 0")

    # ---------------- DEBUG ----------------

    def __repr__(self):
        return (
            f"CacheConfig("
            f"threshold={self.similarity_threshold}, "
            f"semantic={self.enable_semantic}, "
            f"exact={self.enable_exact}, "
            f"metrics={self.enable_metrics}, "
            f"token_cost={self.enable_token_cost}, "
            f"model={self.model_name}, "
            f"top_k={self.top_k}, "
            f"ttl={self.ttl}"
            f")"
        )