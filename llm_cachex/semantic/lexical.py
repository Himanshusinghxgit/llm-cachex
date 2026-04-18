from rank_bm25 import BM25Okapi


class LexicalEngine:
    def __init__(self):
        self.docs = []
        self.ids = []
        self.tokenized = []
        self.bm25 = None

    def _tokenize(self, text):
        return text.lower().split()

    def add(self, text, doc_id):
        self.docs.append(text)
        self.ids.append(doc_id)
        self.tokenized.append(self._tokenize(text))
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query, k=3):
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(self._tokenize(query))

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = []
        for idx, score in ranked:
            norm = score / (score + 10)
            results.append({
                "id": self.ids[idx],
                "score": float(norm)
            })

        return results