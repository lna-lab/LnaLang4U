"""Tests for Anthropic Messages endpoint (via Gateway)."""

import os
import pytest
import requests

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:9010")
HEADERS = {"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"}
MODEL = os.environ.get("ANTHROPIC_MODEL", "anthropic/deepseek-v4-flash")


def gateway_reachable() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def _gw_check():
    if not gateway_reachable():
        pytest.skip("Gateway not reachable")


@pytest.mark.usefixtures("_gw_check")
class TestAnthropicMessages:
    def test_list_models(self):
        r = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "data" in data
        assert len(data["data"]) > 0
        # The default model should be listed
        ids = [m["id"] for m in data["data"]]
        assert "deepseek-v4-flash" in ids

    def test_messages_text(self):
        r = requests.post(f"{BASE_URL}/v1/messages", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 10,
        }, headers=HEADERS, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert len(data["content"][0]["text"]) > 0

    def test_messages_with_system(self):
        r = requests.post(f"{BASE_URL}/v1/messages", json={
            "model": MODEL,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 10,
        }, headers=HEADERS, timeout=60)
        assert r.status_code == 200
        assert len(r.json()["content"][0]["text"]) > 0

    def test_count_tokens(self):
        r = requests.post(f"{BASE_URL}/v1/messages/count_tokens", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello world"}],
        }, headers=HEADERS, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "input_tokens" in data
        assert data["input_tokens"] > 0

    def test_health(self):
        r = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=10)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_unknown_model_fallback(self):
        """Unknown model should fall through to the default model."""
        r = requests.post(f"{BASE_URL}/v1/messages", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }, headers=HEADERS, timeout=60)
        assert r.status_code == 200  # falls back gracefully
