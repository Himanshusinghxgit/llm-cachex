from __future__ import annotations

import time
import threading
from typing import Dict

# OpenAI pricing (can later move to config)
INPUT_COST_PER_1K = 0.00015   # USD
OUTPUT_COST_PER_1K = 0.0006   # USD
USD_TO_INR = 83


class Metrics:
    """
    Tracks cache performance, latency, and cost.
    Thread-safe for concurrent usage.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.hits = 0
        self.misses = 0

        self.llm_latency = []
        self.cache_latency = []

        # real cost tracking
        self.total_cost_usd = 0.0
        self.saved_cost_usd = 0.0

    # ---------------- RECORD ----------------

    def record_hit(self, latency: float, saved_cost_usd: float = 0.0):
        with self._lock:
            self.hits += 1
            self.cache_latency.append(latency)
            self.saved_cost_usd += saved_cost_usd

    def record_miss(
        self,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        with self._lock:
            self.misses += 1
            self.llm_latency.append(latency)

            cost_usd = (
                (input_tokens / 1000) * INPUT_COST_PER_1K +
                (output_tokens / 1000) * OUTPUT_COST_PER_1K
            )

            self.total_cost_usd += cost_usd

    # ---------------- SUMMARY ----------------

    def summary(self) -> Dict:
        with self._lock:
            total = self.hits + self.misses

            hit_rate = (self.hits / total * 100) if total else 0

            avg_llm = (
                sum(self.llm_latency) / len(self.llm_latency)
                if self.llm_latency else 0
            )

            avg_cache = (
                sum(self.cache_latency) / len(self.cache_latency)
                if self.cache_latency else 0
            )

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 2),

                "avg_llm_latency_ms": round(avg_llm * 1000, 2),
                "avg_cache_latency_ms": round(avg_cache * 1000, 2),

                "total_cost_usd": round(self.total_cost_usd, 6),
                "total_cost_rupees": round(self.total_cost_usd * USD_TO_INR, 4),

                "saved_cost_usd": round(self.saved_cost_usd, 6),
                "saved_cost_rupees": round(self.saved_cost_usd * USD_TO_INR, 4),
            }

    # ---------------- RESET ----------------

    def reset(self):
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.llm_latency.clear()
            self.cache_latency.clear()
            self.total_cost_usd = 0.0
            self.saved_cost_usd = 0.0


# global singleton
metrics = Metrics()