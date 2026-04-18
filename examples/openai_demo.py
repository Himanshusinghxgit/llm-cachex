import os
import time

from llm_cachex import llm_cache, metrics

# 👉 LangChain OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 🔐 Set your API key
# export OPENAI_API_KEY="your_key_here"
# or uncomment below:
os.environ["OPENAI_API_KEY"] = ""
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",   # cheap + fast
    temperature=0
)


# 🔥 Wrap with your cache
# @llm_cache
# def call_llm(query):
#     print(">>> REAL LLM CALL")

#     start = time.time()

#     response = llm.invoke([
#         HumanMessage(content=query)
#     ])

#     latency = time.time() - start
#     print(f"LLM latency: {round(latency*1000,2)} ms")

#     return response.content

@llm_cache
def call_llm(query):
    print(">>> REAL LLM CALL")

    response = llm.invoke([
        HumanMessage(content=query)
    ])

    return response.content
# -------------------------
# 🧪 TEST RUN
# -------------------------

print("\n--- FIRST CALL (MISS) ---")
print(call_llm("What is Artificial Intelligence?", user_id="u1", session_id="s1"))

print("\n--- SECOND CALL (SEMANTIC HIT) ---")
print(call_llm("Explain AI in simple terms", user_id="u1", session_id="s1"))

print("\n--- THIRD CALL (EXACT HIT) ---")
print(call_llm("What is Artificial Intelligence?", user_id="u1", session_id="s1"))


# -------------------------
# 📊 METRICS
# -------------------------

print("\n--- METRICS ---")
print(metrics.summary())