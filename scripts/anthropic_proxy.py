#!/usr/bin/env python3
"""Anthropic-to-OpenAI API translation proxy for sglang.

Listens for Anthropic /v1/messages requests, translates them to
OpenAI /v1/chat/completions, forwards to sglang, and translates back.

Usage:
    python3 anthropic_proxy.py [--port 9001] [--target http://127.0.0.1:9000]
"""

import argparse
import json
import logging
import re
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("anthropic-proxy")
app = FastAPI(title="Anthropic Proxy for sglang")

TARGET_URL: str = "http://127.0.0.1:9000"


def extract_system(messages: list) -> tuple[Optional[str], list]:
    """Extract system message from Anthropic messages format."""
    system = None
    remaining = list(messages)
    if remaining and remaining[0].get("role") == "system":
        system = remaining[0].get("content", "")
        remaining = remaining[1:]
    return system, remaining


def translate_messages(anthropic_messages: list) -> list:
    """Convert Anthropic messages to OpenAI format."""
    openai_messages = []
    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Anthropic content can be a list of content blocks
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    # Image support would go here
                    pass
            content = "\n".join(text_parts)

        # Map roles
        if role == "assistant":
            openai_messages.append({"role": "assistant", "content": content})
        else:
            openai_messages.append({"role": "user", "content": content})

    return openai_messages


def build_openai_payload(anthropic_body: dict) -> dict:
    """Build OpenAI /v1/chat/completions payload from Anthropic /v1/messages body."""
    messages = anthropic_body.get("messages", [])
    system, remaining = extract_system(messages)
    openai_messages = translate_messages(remaining)

    if system:
        openai_messages.insert(0, {"role": "system", "content": system})

    payload = {
        "model": anthropic_body.get("model", "deepseek-v4-flash"),
        "messages": openai_messages,
        "max_tokens": anthropic_body.get("max_tokens", 1024),
        "temperature": anthropic_body.get("temperature", 1.0),
        "top_p": anthropic_body.get("top_p", 1.0),
        "stream": anthropic_body.get("stream", False),
    }

    # Top-k
    if "top_k" in anthropic_body:
        payload["top_k"] = anthropic_body["top_k"]

    # Stop sequences
    if "stop_sequences" in anthropic_body:
        payload["stop"] = anthropic_body["stop_sequences"]

    return payload


def translate_response(openai_data: dict, model: str) -> dict:
    """Translate OpenAI chat completion response to Anthropic format."""
    choice = openai_data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = openai_data.get("usage", {})

    anthropic_response = {
        "id": openai_data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": message.get("content", ""),
            }
        ],
        "stop_reason": choice.get("finish_reason", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }

    # Map finish_reason
    finish = choice.get("finish_reason")
    if finish == "stop":
        anthropic_response["stop_reason"] = "end_turn"
    elif finish == "length":
        anthropic_response["stop_reason"] = "max_tokens"
    elif finish == "tool_calls":
        anthropic_response["stop_reason"] = "tool_use"

    return anthropic_response


@app.post("/v1/messages")
async def handle_messages(request: Request):
    body = await request.json()
    model = body.get("model", "deepseek-v4-flash")
    stream = body.get("stream", False)

    # Translate request
    openai_payload = build_openai_payload(body)

    target = f"{TARGET_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }

    if stream:
        # Streaming: forward as SSE, translate on the fly
        openai_payload["stream"] = True
        resp = requests.post(target, json=openai_payload, headers=headers, stream=True)

        async def generate():
            yield f'data: {{"type":"message_start","message":{{"id":"msg_placeholder","type":"message","role":"assistant","model":"{model}","content":[]}}}}' + "\n\n"

            full_text = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            yield f'data: {{"type":"content_block_delta","delta":{{"type":"text_delta","text":{json.dumps(content)}}}}}' + "\n\n"
                    except json.JSONDecodeError:
                        pass

            yield f'data: {{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"output_tokens":0}}}}' + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Non-streaming: translate
        try:
            resp = requests.post(target, json=openai_payload, headers=headers, timeout=300)
            resp.raise_for_status()
            openai_data = resp.json()
            anthropic_response = translate_response(openai_data, model)
            return JSONResponse(content=anthropic_response)
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="Upstream timeout")
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=502, detail="Cannot connect to sglang")
        except requests.exceptions.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"sglang error: {e}")


@app.get("/health")
async def health():
    try:
        resp = requests.get(f"{TARGET_URL}/v1/models", timeout=5)
        return {"status": "ok", "upstream": resp.status_code == 200}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/v1/models")
async def list_models():
    """Forward model list from sglang."""
    try:
        resp = requests.get(f"{TARGET_URL}/v1/models", timeout=10)
        return JSONResponse(content=resp.json())
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


def main():
    global TARGET_URL
    parser = argparse.ArgumentParser(description="Anthropic-to-OpenAI proxy for sglang")
    parser.add_argument("--port", type=int, default=9001, help="Proxy listen port")
    parser.add_argument("--target", type=str, default="http://127.0.0.1:9000", help="sglang target URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    args = parser.parse_args()

    TARGET_URL = args.target.rstrip("/")
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    logger.info(f"Anthropic proxy starting on {args.host}:{args.port} → {TARGET_URL}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
