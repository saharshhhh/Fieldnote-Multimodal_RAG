# eval/ping_cohere.py
import os
from dotenv import load_dotenv
import cohere

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
try:
    co.chat(model="command-a-plus-05-2026",
            max_tokens=5, messages=[{"role": "user", "content": "hi"}])
    print("ok: key works")
except Exception as e:
    print(type(e).__name__, getattr(e, "status_code", None))
    print(getattr(e, "body", e))