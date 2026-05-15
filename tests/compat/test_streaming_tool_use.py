"""Tests for streaming and tool-use behavior."""

import json
import os
import pytest
import requests

OPENAI_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")
GW_URL = os.environ.get("GATEWAY_BASE_URL", "http://127.0.0.1:9010")
MODEL = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")


def any_reachable() -> bool:
    try:
        r = requests.get(f"{OPENAI_URL}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def _check():
    if not any_reachable():
        pytest.skip("No upstream reachable")


@pytest.mark.usefixtures("_check")
class TestStreamingToolUse:
    def test_openai_stream_chunks(self):
        """Each streaming chunk should be valid SSE."""
        r = requests.post(f"{OPENAI_URL}/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Count to 3"}],
            "max_tokens": 20, "stream": True,
        }, stream=True, timeout=60)
        assert r.status_code == 200
        chunks = []
        for line in r.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: ") and decoded[6:] != "[DONE]":
                try:
                    json.loads(decoded[6:])
                    chunks.append(decoded)
                except json.JSONDecodeError:
                    pass
        assert len(chunks) > 0

    def test_openai_tool_call_unsupported(self):
        """If tool calls are not supported, the server should still respond."""
        r = requests.post(f"{OPENAI_URL}/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}}
            }],
            "max_tokens": 20,
        }, timeout=60)
        assert r.status_code == 200  # Should respond gracefully

    def test_anthropic_stream_content(self):
        """Anthropic streaming should produce content_block_delta events."""
        try:
            r = requests.post(f"{GW_URL}/v1/messages", json={
                "model": MODEL, "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20, "stream": True,
            }, headers={"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"},
            stream=True, timeout=60)
            assert r.status_code == 200, f"Gateway returned {r.status_code}"
            text_deltas = 0
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    try:
                        d = json.loads(decoded[6:])
                        if d.get("type") == "content_block_delta":
                            text_deltas += 1
                    except json.JSONDecodeError:
                        pass
            assert text_deltas > 0, "No content_block_delta events received"
        except requests.exceptions.ConnectionError:
            pytest.skip("Gateway not reachable")
