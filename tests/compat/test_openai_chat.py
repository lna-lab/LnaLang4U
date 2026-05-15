"""Tests for OpenAI Chat Completions endpoint (direct SGLang)."""

import os
import subprocess
import pytest
import requests

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")
MODEL = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")


def engine_reachable() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def _engine_check():
    if not engine_reachable():
        pytest.skip("Engine not reachable")


@pytest.mark.usefixtures("_engine_check")
class TestOpenAIChat:
    def test_list_models(self):
        r = requests.get(f"{BASE_URL}/models", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("data", [])) > 0
        assert any(m["id"] == MODEL for m in data["data"])

    def test_chat_non_stream(self):
        r = requests.post(f"{BASE_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 10, "temperature": 0,
        }, timeout=60)
        assert r.status_code == 200
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        assert data["usage"]["completion_tokens"] > 0

    def test_chat_stream(self):
        r = requests.post(f"{BASE_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 10, "temperature": 0, "stream": True,
        }, stream=True, timeout=60)
        assert r.status_code == 200
        chunks = 0
        for line in r.iter_lines():
            if line:
                chunks += 1
        assert chunks > 0

    def test_chat_empty_message(self):
        r = requests.post(f"{BASE_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": ""}],
            "max_tokens": 5, "temperature": 0,
        }, timeout=60)
        assert r.status_code == 200

    def test_chat_model_alias(self):
        """Verify that the default model name resolves correctly."""
        r = requests.post(f"{BASE_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5, "temperature": 0,
        }, timeout=60)
        assert r.status_code == 200
        assert r.json()["model"] == MODEL

    def test_chat_max_tokens_limit(self):
        r = requests.post(f"{BASE_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Write a long essay"}],
            "max_tokens": 5, "temperature": 0,
        }, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert data["usage"]["completion_tokens"] <= 10  # allow small slack

    def test_chat_temperature_zero(self):
        """Deterministic output at temperature=0."""
        r1 = requests.post(f"{BASE_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10, "temperature": 0,
        }, timeout=60)
        r2 = requests.post(f"{BASE_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10, "temperature": 0,
        }, timeout=60)
        assert r1.json()["choices"][0]["message"]["content"] == \
               r2.json()["choices"][0]["message"]["content"]
