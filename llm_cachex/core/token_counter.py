from __future__ import annotations

import logging
from typing import Optional, Union, List

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counter using tiktoken.
    Safe fallback if model encoding not found.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or "gpt-4o-mini"

        try:
            self.enc = tiktoken.encoding_for_model(self.model)
        except Exception:
            logger.warning(f"Model {self.model} not found. Using cl100k_base encoding.")
            self.enc = tiktoken.get_encoding("cl100k_base")

    # ---------------- COUNT ----------------

    def count(self, text: str) -> int:
        if not text:
            return 0

        try:
            return len(self.enc.encode(text))
        except Exception:
            logger.exception("Token counting failed")
            return 0

    def count_batch(self, texts: List[str]) -> int:
        """
        Total tokens across multiple texts
        """
        return sum(self.count(t) for t in texts)