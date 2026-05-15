# API Compatibility

## OpenAI Chat Completions (native)

Provided directly by SGLang. No gateway required.

**Endpoint:** `POST /v1/chat/completions`
**Base URL:** `http://<host>:9000`
**Model:** `deepseek-v4-flash`

### Supported

- ✅ Text messages
- ✅ Streaming (`stream: true`)
- ✅ Tool/function calling
- ✅ `max_tokens`, `temperature`, `top_p`, `stop`
- ✅ `GET /v1/models`

### Example

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

### Python

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

## Anthropic Messages API (via Gateway)

Provided by the LnaLang4U Gateway. Translates Anthropic-style requests to OpenAI Chat Completions.

**Endpoint:** `POST /v1/messages`
**Base URL:** `http://<host>:9010`
**Model:** `anthropic/deepseek-v4-flash`

### Supported

- ✅ Text messages
- ✅ System prompts
- ✅ Streaming
- ✅ Token counting (`/v1/messages/count_tokens`)
- ✅ `GET /v1/models`

### Experimental

- 🧪 Tool use
- 🧪 Multi-turn conversations

### Unsupported

- ❌ Thinking blocks (not implemented yet)

### Example

```bash
curl http://127.0.0.1:9010/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-dev" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "anthropic/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

### Token counting

```bash
curl http://127.0.0.1:9010/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-dev" \
  -d '{
    "model": "anthropic/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Returns approximate token count. Token counting is approximate unless the DeepSeek-V4-Flash tokenizer is available inside the gateway container.

## OpenAI Responses API (via Gateway)

Provided by the LnaLang4U Gateway. Translates Responses API requests to Chat Completions.

**Endpoint:** `POST /v1/responses`
**Base URL:** `http://<host>:9010`
**Model:** `codex-lnalang4u`

### Supported

- ✅ Text input (string and message-list)
- ✅ System instructions
- ✅ `max_output_tokens`, `temperature`

### Experimental

- 🧪 Streaming
- 🧪 Tool calls

### Example

```bash
curl http://127.0.0.1:9010/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codex-lnalang4u",
    "input": "Hello",
    "max_output_tokens": 100
  }'
```

## Model aliases

The Gateway maintains model aliases for client compatibility:

| Alias | Resolves to |
|-------|-------------|
| `deepseek-v4-flash` | deepseek-v4-flash |
| `anthropic/deepseek-v4-flash` | deepseek-v4-flash |
| `claude-lnalang4u` | deepseek-v4-flash |
| `codex-lnalang4u` | deepseek-v4-flash |

All aliases serve the same underlying model. No additional models are available.
