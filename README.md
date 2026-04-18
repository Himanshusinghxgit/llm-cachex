# 🚀 llm-cachex

**Drop-in caching + retrieval layer for LLM applications (RAG, agents, chatbots).**

Stop paying for repeated LLM calls.
Automatically reuse responses using **exact + semantic caching** with zero changes to your business logic.

---

## ✨ Why llm-cachex?

Most LLM apps repeatedly call the model for:

* Slightly rephrased questions
* Agent/tool loops
* Chat history variations

👉 This wastes **latency + money**

**llm-cachex fixes that automatically.**

---

## 🔥 Features

* ⚡ **Exact cache** (Redis-backed)
* 🧠 **Semantic cache** (FAISS + embeddings)
* 🔍 **Hybrid retrieval** (BM25 + vector search)
* 🧬 **Cross-encoder reranking** (high-quality matches)
* 🤖 **Agent + tool support**
* 🧵 **Memory-aware context support**
* 💰 **Token + cost tracking**
* 🧩 **Plug-and-play decorator API**

---

## 🏗️ Architecture

```text
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

## 📦 Installation

```bash
pip install -e .
```

(For now, install locally. PyPI support coming soon.)

---

## 🚀 Quick Start

```python
from llm_cachex import llm_cache, CacheConfig

@llm_cache(CacheConfig())
def ask_llm(prompt):
    return llm(prompt)

print(ask_llm("What is AI?"))        # LLM call
print(ask_llm("Explain AI"))         # Semantic cache hit
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

## 🧠 Semantic Cache (What makes this powerful)

Unlike basic caching, this system:

```text
"What is AI?"
"Explain artificial intelligence"
```

👉 returns cached answer (no LLM call)

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

Example:

```text
{
  'hits': 2,
  'misses': 1,
  'hit_rate': 66.67,
  'avg_llm_latency_ms': 2000,
  'avg_cache_latency_ms': 30,
  'total_cost_rupees': 0.01
}
```

---

## 🧪 Examples

Run demos:

```bash
python examples/basic.py
python examples/rag_demo.py
python examples/agent_demo.py
python examples/strict_test.py
```

---

## 📁 Project Structure

```text
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

PRs welcome. Open an issue for discussions.

---

## 📜 License

MIT License

---

## 👤 Author

Himanshu Singh

---

## ⭐ If this helps you

Give a star. It helps the project grow.
