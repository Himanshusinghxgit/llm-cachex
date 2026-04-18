from __future__ import annotations

import logging
from typing import List, Dict, Optional

from llm_cachex.embedding.embedder import Embedder
from llm_cachex.index.faiss_index import FAISSIndex

from sentence_transformers import CrossEncoder
from llm_cachex.semantic.lexical import LexicalEngine

logger = logging.getLogger(__name__)


class SemanticEngine:
    """
    Combines embedding + vector index.
    Responsible for semantic retrieval only.
    """



    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        index: Optional[FAISSIndex] = None,
    ):
        self.embedder = embedder or Embedder()
        self.index = index or FAISSIndex()

        # 🔥 NEW
        self.lexical = LexicalEngine()
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # store text for reranker
        self.id_to_text = {}
    # ---------------- ADD ----------------

    def add(self, text: str, doc_id: str):
        try:
            vector = self.embedder.encode(text)
            self.index.add(vector, doc_id)

            # 🔥 NEW
            self.lexical.add(text, doc_id)
            # self.id_to_text[doc_id] = text
            self.id_to_text[doc_id] = {
                "query": text,
                "response": None
            }

        except Exception:
            logger.exception("Semantic add failed")

    def add_batch(self, texts: List[str], ids: List[str]):
        """
        Batch insert
        """
        try:
            vectors = self.embedder.encode(texts)
            self.index.add_batch(vectors, ids)

        except Exception:
            logger.exception("Semantic batch add failed")

    # ---------------- SEARCH ----------------

    def search(self, query: str, k: int = 3) -> List[Dict]:
        try:
            # 🔹 semantic search
            vector = self.embedder.encode(query)
            sem_results = self.index.search(vector, k=k)

            # 🔹 lexical search
            lex_results = self.lexical.search(query, k=k)

            # 🔹 combine scores
            combined = {}

            for r in sem_results:
                combined[r["id"]] = r["score"] * 0.6

            for r in lex_results:
                combined[r["id"]] = combined.get(r["id"], 0) + r["score"] * 0.4

            if not combined:
                return []

            # 🔹 prepare reranker
            candidates = list(combined.keys())
            # pairs = [(query, self.id_to_text[cid]) for cid in candidates]
            pairs = [
                (query, self.id_to_text[cid]["response"] or self.id_to_text[cid]["query"])
                for cid in candidates
            ]

            rerank_scores = self.reranker.predict(pairs)

            # 🔹 final ranking
            results = []
            for i, cid in enumerate(candidates):
                results.append({
                    "id": cid,
                    "score": float(rerank_scores[i])
                })

            results.sort(key=lambda x: x["score"], reverse=True)

            return results[:k]

        except Exception:
            logger.exception("Semantic search failed")
            return []

    # ---------------- SAVE ----------------

    def save(self):
        """
        Persist FAISS index
        """
        try:
            self.index.save()

        except Exception:
            logger.exception("Semantic index save failed")