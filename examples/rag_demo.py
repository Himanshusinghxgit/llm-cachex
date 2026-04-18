from pathlib import Path
import os
from dotenv import load_dotenv

# Load env
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from llm_cachex import llm_cache, CacheConfig, metrics

client = OpenAI()

# 🔹 Fake knowledge base
DOCUMENTS = [
    "AI is the simulation of human intelligence.",
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
]


def retrieve(query):
    # naive retriever
    return " ".join(DOCUMENTS[:2])


@llm_cache(
    CacheConfig(
        model_name="gpt-4o-mini",
        similarity_threshold=0.75,
        top_k=5
    )
)
def rag_pipeline(query: str):
    context = retrieve(query)

    final_prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

    print(">>> RAG LLM CALLED")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("\n--- FIRST ---")
    print(rag_pipeline("What is AI?"))

    print("\n--- SEMANTIC ---")
    print(rag_pipeline("Explain artificial intelligence"))

    print("\n--- EXACT ---")
    print(rag_pipeline("What is AI?"))

    print("\nMETRICS:")
    print(metrics.summary())