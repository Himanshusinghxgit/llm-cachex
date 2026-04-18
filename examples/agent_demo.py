from pathlib import Path
from dotenv import load_dotenv
import os

# load .env from this folder
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from llm_cachex import llm_cache, CacheConfig, metrics

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 🔹 fake tools
def tool_search(q):
    return f"[TOOL SEARCH RESULT] {q}"


def tool_calc(expr):
    return str(eval(expr))


@llm_cache(
    CacheConfig(
        model_name="gpt-4o-mini",
        similarity_threshold=0.65,   # not heavily used now (reranker dominates)
        top_k=3
    )
)
def agent(raw_query: str, full_prompt: str):
    print(">>> AGENT EXECUTION")

    if "calculate" in raw_query:
        return str(eval(raw_query.replace("calculate", "").strip()))

    if "search" in raw_query:
        return f"[TOOL SEARCH RESULT] {raw_query}"

    # fallback to LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )

    return response.choices[0].message.content
    
if __name__ == "__main__":
    print("\n--- TOOL CALL (MISS) ---")
    print(agent("calculate 10 + 5"))

    print("\n--- EXACT HIT ---")
    print(agent("calculate 10 + 5"))

    print("\n--- SEARCH (MISS) ---")
    print(agent("search AI trends"))

    print("\n--- SEMANTIC HIT ---")
    print(agent("search artificial intelligence trends"))

    print("\nMETRICS:")
    print(metrics.summary())