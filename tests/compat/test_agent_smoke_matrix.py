"""Smoke matrix: runs basic tests across all agent paths."""

import os
import pytest
import requests

OPENAI_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")
GW_URL = os.environ.get("GATEWAY_BASE_URL", "http://127.0.0.1:9010")
MODEL = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")


def _ok(url, **kwargs):
    try:
        r = requests.get(url, timeout=5, **kwargs)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ok(f"{OPENAI_URL}/models"), reason="Engine not reachable")
def test_openai_sdk_direct():
    r = requests.post(f"{OPENAI_URL}/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5, "temperature": 0,
    }, timeout=60)
    assert r.status_code == 200


@pytest.mark.skipif(not _ok(f"{GW_URL}/health"), reason="Gateway not reachable")
def test_openai_sdk_gateway():
    r = requests.post(f"{GW_URL}/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5, "temperature": 0,
    }, timeout=60)
    assert r.status_code == 200


@pytest.mark.skipif(not _ok(f"{GW_URL}/health"), reason="Gateway not reachable")
def test_claude_code_gateway():
    r = requests.post(f"{GW_URL}/v1/messages", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }, headers={"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"}, timeout=60)
    assert r.status_code == 200


@pytest.mark.skipif(not _ok(f"{GW_URL}/health"), reason="Gateway not reachable")
def test_codex_gateway():
    r = requests.post(f"{GW_URL}/v1/responses", json={
        "model": MODEL, "input": "hi", "max_output_tokens": 5,
    }, timeout=60)
    assert r.status_code == 200
