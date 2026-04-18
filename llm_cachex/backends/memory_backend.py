# llm_cachex/backends/memory_backend.py

class MemoryBackend:
    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def exists(self, key):
        return key in self.store

    def add(self, query, answer):
        query_hash = get_query_hash(query)

        # ❌ Already exists → skip
        if self.backend.exists(query_hash):
            return

        embedding = self.embedder.encode(query)

        self.backend.set(query_hash, {
            "query": query,
            "answer": answer
        })

        self.index.add(embedding, query_hash)
        semantic_engine.id_to_text[query_hash]["response"] = response

    def set(self, key, value):
        self.store[key] = value

    # def get(self, key):
    #     return self.store.get(key)

    # def exists(self, key):
    #     return key in self.store

    def get_all(self):
        return self.store