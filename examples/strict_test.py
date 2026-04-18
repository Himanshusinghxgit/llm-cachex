from llm_cachex import llm_cache, CacheConfig, metrics


@llm_cache(
    CacheConfig(
        enable_semantic=True,
        enable_exact=True,
        similarity_threshold=0.9,  # VERY strict
        enable_token_cost=False
    )
)
def test_fn(query):
    print(">>> LLM CALLED")
    return f"Response for: {query}"


if __name__ == "__main__":
    print("\n--- FIRST ---")
    print(test_fn("What is AI?"))

    print("\n--- SHOULD MISS (STRICT) ---")
    print(test_fn("Explain AI"))   # semantic SHOULD NOT hit

    print("\n--- EXACT HIT ---")
    print(test_fn("What is AI?"))

    print("\nMETRICS:")
    print(metrics.summary())