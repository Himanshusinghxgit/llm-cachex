from __future__ import annotations

import hashlib
import logging
from typing import Optional

import redis

from llm_cachex.semantic.engine import SemanticEngine

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Single source of truth for:
    - Exact cache (Redis / memory)
    - Semantic engine (hybrid retrieval)
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        ttl: int = 3600,
    ):
        self.ttl = ttl

        # 🔥 SINGLE semantic engine (shared everywhere)
        self.semantic_engine = SemanticEngine()

        try:
            self.store = redis_client or redis.Redis(
                host="localhost",
                port=6379,
                decode_responses=True,
            )
            self.store.ping()
        except Exception:
            logger.warning("Redis unavailable. Falling back to memory.")
            self.store = {}

    # ---------------- EXACT CACHE ----------------

    def get(self, user_id: str, query: str) -> Optional[str]:
        key = self._hash(user_id, query)

        try:
            if isinstance(self.store, dict):
                return self.store.get(key)

            value = self.store.get(key)
            if value:
                logger.debug("[EXACT CACHE HIT]")
            return value

        except Exception:
            logger.exception("Cache get failed")
            return None

    def set(self, user_id: str, query: str, response: str, ttl: Optional[int] = None):
        ttl = ttl or self.ttl
        key = self._hash(user_id, query)

        try:
            # 🔹 store in Redis / memory
            if isinstance(self.store, dict):
                self.store[key] = response
            else:
                self.store.set(key, response, ex=ttl)

            # 🔥 IMPORTANT: add AFTER response exists
            self.semantic_engine.add(query, key)

            # 🔥 store full object for reranker
            self.semantic_engine.id_to_text[key] = {
                "query": query,
                "response": response,
            }

        except Exception:
            logger.exception("Cache set failed")

    # ---------------- DIRECT ACCESS ----------------

    def get_by_id(self, key: str) -> Optional[str]:
        try:
            if isinstance(self.store, dict):
                return self.store.get(key)

            return self.store.get(key)

        except Exception:
            logger.exception("Cache get_by_id failed")
            return None

    # ---------------- HASH ----------------

    def _hash(self, user_id: str, query: str) -> str:
        raw = f"{user_id}:{query}"
        return hashlib.sha256(raw.encode()).hexdigest()