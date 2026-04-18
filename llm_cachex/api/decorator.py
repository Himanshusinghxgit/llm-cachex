from __future__ import annotations

import time
import logging
import asyncio
import inspect
from typing import Callable, Any, Optional

from llm_cachex.config import CacheConfig
from llm_cachex.core.metrics import metrics
from llm_cachex.core.token_counter import TokenCounter
from llm_cachex.core.cache_manager import CacheManager
from llm_cachex.core.memory import ChatMemory

logger = logging.getLogger(__name__)


def llm_cache(
    config: Optional[CacheConfig] = None,
    cache: Optional[CacheManager] = None,
    memory: Optional[ChatMemory] = None,
) -> Callable:

    config = config or CacheConfig()
    cache = cache or CacheManager()
    memory = memory or ChatMemory()

    def decorator(func: Callable):

        # 🔥 detect ONCE (never inside try)
        is_agent = len(inspect.signature(func).parameters) == 2

        def sync_wrapper(
            query: str,
            user_id: str = "default",
            session_id: str = "default"
        ) -> Any:

            start = time.time()

            # ---------------- EXACT CACHE ----------------
            if config.enable_exact:
                cached = cache.get(user_id, query)
                if cached:
                    _record_hit(start)
                    return cached

            # ---------------- SEMANTIC CACHE ----------------
            if config.enable_semantic:
                results = cache.semantic_engine.search(query, k=config.top_k)

                if results:
                    best = results[0]
                    value = cache.get_by_id(best["id"])

                    print("DEBUG BEST SCORE:", best["score"])

                    if value:
                        # 🔥 tool-safe bypass (no reranker needed)
                        if isinstance(value, str) and value.startswith("[TOOL"):
                            _record_hit(start)
                            return value

                        # 🔥 reranker threshold
                        if best["score"] > 0.2:
                            _record_hit(start)
                            return value

            # ---------------- MEMORY ----------------
            history = memory.get(user_id, session_id) if memory else []

            context = "".join(
                f"User: {h['q']}\nAssistant: {h['a']}\n"
                for h in history
            )

            raw_query = query
            final_query = context + f"User: {query}"

            logger.info(f"[LLM CALL] → {query}")

            # ---------------- TOKEN COUNT ----------------
            token_counter = (
                TokenCounter(config.model_name)
                if config.enable_token_cost and config.model_name
                else None
            )

            input_tokens = token_counter.count(final_query) if token_counter else 0

            # ---------------- EXECUTION ----------------
            start_llm = time.time()

            try:
                if is_agent:
                    response = func(raw_query, final_query)
                else:
                    response = func(final_query)

            except Exception:
                logger.exception("LLM execution failed — fallback")

                # 🔥 fallback (safe, no recursion loop)
                if is_agent:
                    response = func(raw_query, raw_query)
                else:
                    response = func(raw_query)

            latency = time.time() - start_llm

            response_text = _normalize_response(response)

            output_tokens = token_counter.count(response_text) if token_counter else 0

            # ---------------- METRICS ----------------
            if config.enable_metrics:
                metrics.record_miss(
                    latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

            # ---------------- STORE ----------------
            cache.set(user_id, query, response_text, ttl=config.ttl)

            if memory:
                memory.add(user_id, session_id, query, response_text)

            return response

        async def async_wrapper(query, user_id="default", session_id="default"):
            return sync_wrapper(query, user_id, session_id)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ================= HELPERS =================

def _record_hit(start_time: float):
    latency = time.time() - start_time
    metrics.record_hit(latency)


def _normalize_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    return str(response)