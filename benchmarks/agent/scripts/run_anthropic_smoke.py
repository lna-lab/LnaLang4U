#!/usr/bin/env python3
"""Smoke test for Anthropic-compatible endpoint (via Gateway)."""
import requests, sys

BASE = "http://127.0.0.1:9010"
HEADERS = {"Content-Type": "application/json", "x-api-key": "local-dev", "anthropic-version": "2023-06-01"}
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
    r = requests.get(f"{BASE}/v1/models", headers=HEADERS, timeout=10)
    check("/v1/models", r.status_code == 200, str(r.status_code))
except Exception as e:
    check("/v1/models", False, str(e))

# POST /v1/messages (non-streaming)
try:
    r = requests.post(f"{BASE}/v1/messages", json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
    }, headers=HEADERS, timeout=60)
    data = r.json()
    content = data.get("content", [{}])[0].get("text", "")
    check("/v1/messages (non-stream)", r.status_code == 200 and len(content) > 0,
          f"status={r.status_code} content={content[:30]}")
except Exception as e:
    check("/v1/messages (non-stream)", False, str(e))

# POST /v1/messages/count_tokens
try:
    r = requests.post(f"{BASE}/v1/messages/count_tokens", json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hello"}],
    }, headers=HEADERS, timeout=10)
    data = r.json()
    check("/v1/messages/count_tokens", r.status_code == 200 and "input_tokens" in data,
          f"status={r.status_code}")
except Exception as e:
    check("/v1/messages/count_tokens", False, str(e))

# GET /health
try:
    r = requests.get(f"{BASE}/health", timeout=10)
    data = r.json()
    check("/health", r.status_code == 200 and data.get("status") == "ok",
          f"status={r.status_code}")
except Exception as e:
    check("/health", False, str(e))

print(f"\nResults: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
