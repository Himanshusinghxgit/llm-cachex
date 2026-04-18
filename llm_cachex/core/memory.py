from __future__ import annotations

import json
import logging
from typing import List, Dict, Optional

import redis

logger = logging.getLogger(__name__)


class ChatMemory:
    """
    Pluggable chat memory.
    Uses Redis if available, else falls back to in-memory.
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        ttl: int = 3600,
    ):
        self.ttl = ttl

        try:
            self.store = redis_client or redis.Redis(
                host="localhost",
                port=6379,
                decode_responses=True
            )
            self.store.ping()
            self._is_redis = True

        except Exception:
            logger.warning("Redis unavailable. Using in-memory chat storage.")
            self.store = {}
            self._is_redis = False

    # ---------------- KEY ----------------

    def _key(self, user_id: str, session_id: str) -> str:
        return f"chat:{user_id}:{session_id}"

    # ---------------- ADD ----------------

    def add(self, user_id: str, session_id: str, query: str, response: str):
        key = self._key(user_id, session_id)

        entry = json.dumps({
            "q": query,
            "a": response
        })

        try:
            if self._is_redis:
                self.store.rpush(key, entry)
                self.store.expire(key, self.ttl)
            else:
                self.store.setdefault(key, []).append(entry)

        except Exception:
            logger.exception("Memory add failed")

    # ---------------- GET ----------------

    def get(
        self,
        user_id: str,
        session_id: str,
        limit: int = 5
    ) -> List[Dict]:

        key = self._key(user_id, session_id)

        try:
            if self._is_redis:
                items = self.store.lrange(key, -limit, -1)
            else:
                items = self.store.get(key, [])[-limit:]

            return [json.loads(i) for i in items]

        except Exception:
            logger.exception("Memory fetch failed")
            return []