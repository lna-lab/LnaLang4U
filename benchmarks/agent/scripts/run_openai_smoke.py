#!/usr/bin/env python3
"""Smoke test for OpenAI-compatible endpoint."""
import requests, json, sys

BASE = "http://127.0.0.1:9000"
passed = 0
failed = 0

def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {detail}")

# GET /v1/models
try:
    r = requests.get(f"{BASE}/v1/models", timeout=10)
    data = r.json()
    check("/v1/models", r.status_code == 200 and len(data.get("data", [])) > 0, str(r.status_code))
except Exception as e:
    check("/v1/models", False, str(e))

# POST /v1/chat/completions (non-streaming)
try:
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10, "temperature": 0,
    }, timeout=60)
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    check("/v1/chat/completions (non-stream)", r.status_code == 200 and len(content) > 0,
          f"status={r.status_code} content={content[:30]}")
except Exception as e:
    check("/v1/chat/completions (non-stream)", False, str(e))

# POST /v1/chat/completions (streaming)
try:
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10, "temperature": 0, "stream": True,
    }, stream=True, timeout=60)
    chunks = 0
    for line in r.iter_lines():
        if line:
            chunks += 1
    check("/v1/chat/completions (stream)", r.status_code == 200 and chunks > 0,
          f"status={r.status_code} chunks={chunks}")
except Exception as e:
    check("/v1/chat/completions (stream)", False, str(e))

print(f"\nResults: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
