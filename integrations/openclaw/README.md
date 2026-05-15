# OpenClaw integration

OpenClaw can connect to LnaLang4U through the OpenAI-compatible SGLang endpoint.

## Endpoint

```text
Base URL: http://127.0.0.1:9000/v1
Model: deepseek-v4-flash
API: OpenAI Chat Completions
Context window: 1048576
```

## Notes

- Direct mode uses SGLang `/v1/chat/completions`.
- Gateway mode is optional.
- Status: ✅ Tested — basic requests verified.
