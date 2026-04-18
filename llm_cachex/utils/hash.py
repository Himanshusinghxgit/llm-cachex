import hashlib

def get_query_hash(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()