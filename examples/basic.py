from pathlib import Path
import os

from dotenv import load_dotenv

# rm -rf faiss.index faiss.index.meta
# redis-cli FLUSHALL

# 🔹 Load .env (from examples folder)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# 🔹 Validate API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# 🔹 Imports after env setup
from openai import OpenAI
from llm_cachex import llm_cache, CacheConfig, metrics

client = OpenAI(api_key=api_key)


@llm_cache(
    CacheConfig(
        model_name="gpt-4o-mini",
        similarity_threshold=0.7,
        top_k=3
    )
)
def ask_llm(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask_llm("What is Artificial Intelligence?"))
    print(ask_llm("Explain AI"))
    print(ask_llm("What is Artificial Intelligence?"))  # cache hit

    print("\nMETRICS:")
    print(metrics.summary())