# 🚀 llmcachex-ai

![PyPI](https://img.shields.io/pypi/v/llmcachex-ai)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Drop-in semantic + exact caching layer for LLM applications (RAG, agents, chatbots)

Save up to **80% LLM cost** and reduce latency by avoiding repeated model calls using intelligent caching.

---

## ⚡ Installation

```bash
pip install llmcachex-ai
```

---

## ✨ Why llmcachex-ai?

Most LLM applications repeatedly call the model for:

* Slightly rephrased queries
* Agent/tool loops
* Chat history variations

This leads to higher latency and unnecessary cost.

**llmcachex-ai solves this automatically** by caching responses intelligently using exact + semantic matching.

---

## 🔥 Features

* ⚡ Exact cache (Redis-backed)
* 🧠 Semantic cache (FAISS + embeddings)
* 🔍 Hybrid retrieval (BM25 + vector search)
* 🧬 Cross-encoder reranking (high-quality matches)
* 🤖 Works with agents and tools
* 🧵 Memory-aware context support
* 💰 Token usage and cost tracking
* 🧩 Plug-and-play decorator API

---

## 🏗️ How It Works

```
User Query
   ↓
llm_cache decorator
   ├── Exact Cache (Redis)
   ├── Semantic Engine
   │     ├── FAISS (vector)
   │     ├── BM25 (lexical)
   │     └── CrossEncoder (rerank)
   └── LLM / Agent
```

---

## 🚀 Quick Start

```python
from llm_cachex import llm_cache, CacheConfig

@llm_cache(CacheConfig())
def ask_llm(prompt):
    return llm(prompt)

print(ask_llm("What is AI?"))      # LLM call
print(ask_llm("Explain AI"))       # Semantic cache hit
```

---

## 🤖 Agent Example

Works seamlessly with tools:

```python
@llm_cache(CacheConfig())
def agent(raw_query, full_prompt):

    if "calculate" in raw_query:
        return str(eval(raw_query.replace("calculate", "").strip()))

    if "search" in raw_query:
        return f"[TOOL SEARCH RESULT] {raw_query}"

    return llm(full_prompt)
```

---

## 🧠 Semantic Cache (Why it’s powerful)

```
"What is AI?"
"Explain artificial intelligence"
```

Both return the same cached response — no additional LLM call required.

---

## ⚙️ Configuration

```python
CacheConfig(
    enable_exact=True,
    enable_semantic=True,
    similarity_threshold=0.7,
    top_k=3,
    model_name="gpt-4o-mini",
    enable_metrics=True,
    enable_token_cost=True
)
```

---

## 📊 Metrics

```python
from llm_cachex import metrics

print(metrics.summary())
```

Example output:

```python
{
  "hits": 2,
  "misses": 1,
  "hit_rate": 66.67,
  "avg_llm_latency_ms": 2000,
  "avg_cache_latency_ms": 30,
  "total_cost_rupees": 0.01
}
```

---

## 🎯 Use Cases

* RAG pipelines
* AI agents & tool execution
* Chatbots with memory
* Cost optimization for LLM APIs
* High-frequency query systems

---

## ⚡ Performance Impact

Typical improvements:

* 2–10x latency reduction
* 50–80% cost savings

---

## 📁 Project Structure

```
llm_cachex/
├── api/            # decorator layer
├── core/           # cache, metrics, memory
├── semantic/       # hybrid search + reranker
├── embedding/      # embeddings
├── index/          # FAISS index
├── similarity/     # similarity utils
├── utils/          # helpers
```

---

## 🧭 Roadmap

* [ ] Async support
* [ ] Streaming support
* [ ] Batch inference
* [ ] Multi-model caching
* [ ] Pluggable vector DBs (Chroma / Pinecone)
* [ ] Observability dashboard

---

## 🤝 Contributing

Contributions are welcome. Open an issue to discuss ideas or submit a PR.

---

## 📜 License

MIT License

---

## 👤 Author

Himanshu Singh

---

## ⭐ Support

If this project helps you, consider giving it a star ⭐ on GitHub.
