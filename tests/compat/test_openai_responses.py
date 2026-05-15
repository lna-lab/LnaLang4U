"""Tests for OpenAI Responses API endpoint (via Gateway)."""

import os
import pytest
import requests

BASE_URL = os.environ.get("RESPONSES_BASE_URL", "http://127.0.0.1:9010")
MODEL = os.environ.get("RESPONSES_MODEL", "codex-lnalang4u")


def gateway_reachable() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def _gw_check():
    if not gateway_reachable():
        pytest.skip("Gateway not reachable")


@pytest.mark.usefixtures("_gw_check")
class TestOpenAIResponses:
    def test_responses_text_input(self):
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": MODEL, "input": "Say OK", "max_output_tokens": 10,
        }, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert len(data["output"]) > 0

    def test_responses_message_input(self):
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": MODEL,
            "input": [{"role": "user", "content": "Say OK"}],
            "max_output_tokens": 10,
        }, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "completed"
        assert len(data["output"]) > 0

    def test_responses_with_instructions(self):
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": MODEL,
            "input": "Say OK",
            "instructions": "You are a helpful assistant.",
            "max_output_tokens": 10,
        }, timeout=60)
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_responses_temperature(self):
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": MODEL, "input": "Say OK",
            "max_output_tokens": 10, "temperature": 0,
        }, timeout=60)
        assert r.status_code == 200

    def test_responses_usage(self):
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": MODEL, "input": "Say OK", "max_output_tokens": 5,
        }, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert "usage" in data
        assert data["usage"]["output_tokens"] > 0
