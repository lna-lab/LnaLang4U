#!/usr/bin/env python3
"""LnaLang4U Gateway — OpenAI / Anthropic / Responses compatibility layer."""

import json
import logging
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .model_registry import get_models, get_anthropic_models

logger = logging.getLogger("lna-gateway")
app = FastAPI(title="LnaLang4U Gateway", version="0.1.0")

UPSTREAM_HEADERS = {
    "Content-Type": "application/json",
}


_client: httpx.AsyncClient = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=300.0)
    return _client


async def proxy_to_upstream(payload: dict, stream: bool = False):
    """Forward request to SGLang upstream."""
    url = f"{settings.upstream_base_url}/chat/completions"
    if stream:
        payload["stream"] = True
    client = await get_client()
    if stream:
        resp = await client.post(url, json=payload, headers=UPSTREAM_HEADERS)
        return resp
    else:
        resp = await client.post(url, json=payload, headers=UPSTREAM_HEADERS)
        return resp


def resolve_model(model: Optional[str]) -> str:
    return settings.resolve_model(model)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health():
    try:
        client = await get_client()
        r = await client.get(f"{settings.upstream_base_url}/models")
        upstream_ok = r.status_code == 200
    except Exception:
        upstream_ok = False
    return {
        "status": "ok" if upstream_ok else "degraded",
        "upstream": settings.upstream_base_url,
        "model": settings.served_model,
        "upstream_connected": upstream_ok,
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": get_models()}


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions (pass-through)
# --------------------------------------------------------------------------- #

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    body["model"] = resolve_model(body.get("model"))
    stream = body.get("stream", False)
    resp = await proxy_to_upstream(body, stream=stream)

    if stream:
        async def generate():
            async for chunk in resp.aiter_bytes():
                yield chunk
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        return JSONResponse(content=resp.json())


# --------------------------------------------------------------------------- #
# Anthropic Messages
# --------------------------------------------------------------------------- #

def _translate_anthropic_to_openai(body: dict) -> dict:
    """Convert Anthropic Messages request to OpenAI Chat Completions."""
    messages = body.get("messages", [])
    system = None
    remaining = list(messages)

    if remaining and remaining[0].get("role") == "system":
        system = remaining[0].get("content", "")
        remaining = remaining[1:]

    openai_messages = []
    for msg in remaining:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
            content = text
        mapped = "assistant" if role == "assistant" else "user"
        openai_messages.append({"role": mapped, "content": content})

    if system:
        openai_messages.insert(0, {"role": "system", "content": system})

    return {
        "model": resolve_model(body.get("model")),
        "messages": openai_messages,
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
        "stream": body.get("stream", False),
        "stop": body.get("stop_sequences"),
    }


def _translate_openai_to_anthropic(openai_data: dict, model: str) -> dict:
    """Convert OpenAI Chat Completion response to Anthropic Messages format."""
    choice = openai_data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    usage = openai_data.get("usage", {})
    finish = choice.get("finish_reason")
    stop_reason = "end_turn" if finish == "stop" else "max_tokens" if finish == "length" else finish

    return {
        "id": openai_data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": msg.get("content", "")}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    model = body.get("model", settings.served_model)

    openai_payload = _translate_anthropic_to_openai(body)

    if stream:
        resp = await proxy_to_upstream(openai_payload, stream=True)
        async def generate():
            yield f'data: {{"type":"message_start","message":{{"id":"msg_1","type":"message","role":"assistant","model":"{model}","content":[]}}}}' + "\n\n"
            full = ""
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        d = json.loads(data)
                        delta = d.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full += content
                            yield f'data: {{"type":"content_block_delta","delta":{{"type":"text_delta","text":{json.dumps(content)}}}}}' + "\n\n"
                    except json.JSONDecodeError:
                        pass
            yield f'data: {{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"output_tokens":0}}}}' + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        resp = await proxy_to_upstream(openai_payload)
        openai_data = resp.json()
        anthropic_resp = _translate_openai_to_anthropic(openai_data, model)
        return JSONResponse(content=anthropic_resp)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    total = sum(len(m.get("content", "")) // 4 for m in messages)
    return {"input_tokens": max(1, total)}


# --------------------------------------------------------------------------- #
# OpenAI Responses API
# --------------------------------------------------------------------------- #

def _translate_responses_to_chat(body: dict) -> dict:
    """Convert Responses API request to Chat Completions."""
    inp = body.get("input", "")
    if isinstance(inp, list):
        inp = " ".join(m.get("content", "") for m in inp if isinstance(m, dict))

    messages = [{"role": "user", "content": inp}]
    instructions = body.get("instructions")
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    return {
        "model": resolve_model(body.get("model")),
        "messages": messages,
        "max_tokens": body.get("max_output_tokens", 1024),
        "temperature": body.get("temperature", 1.0),
        "stream": body.get("stream", False),
    }


def _translate_chat_to_responses(chat_data: dict, model: str) -> dict:
    """Convert Chat Completion response to Responses API format."""
    choice = chat_data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    usage = chat_data.get("usage", {})

    return {
        "id": chat_data.get("id", ""),
        "object": "response",
        "model": model,
        "created": int(time.time()),
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": msg.get("content", "")}]}],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


@app.post("/v1/responses")
async def openai_responses(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    model = body.get("model", settings.served_model)

    openai_payload = _translate_responses_to_chat(body)

    if stream:
        resp = await proxy_to_upstream(openai_payload, stream=True)
        async def generate():
            async for line in resp.aiter_lines():
                yield line + "\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        resp = await proxy_to_upstream(openai_payload)
        chat_data = resp.json()
        responses_data = _translate_chat_to_responses(chat_data, model)
        return JSONResponse(content=responses_data)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info(f"Gateway starting on {settings.host}:{settings.port} → {settings.upstream_base_url}")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
