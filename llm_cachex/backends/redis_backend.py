# llm_cachex/backends/redis_backend.py

import redis
import json
import numpy as np


class RedisBackend:
    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def add(self, query, embedding, response):
        key = f"cache:{query}"

        value = json.dumps({
            "embedding": embedding.tolist(),
            "response": response
        })

        self.client.set(key, value)

    def get_all(self):
        keys = self.client.keys("cache:*")

        results = []
        for key in keys:
            data = json.loads(self.client.get(key))
            results.append({
                "query": key.decode().replace("cache:", ""),
                "embedding": np.array(data["embedding"]),
                "response": data["response"]
            })

        return results
    
    def exists(self, key):
        return self.client.exists(key)
    
    def exists(self, key):
        return key in self.store