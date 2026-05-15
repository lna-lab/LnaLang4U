# Agent Compatibility

LnaLang4U can be used as a local long-context backend for coding agents and agent frameworks.

## Supported API surfaces

- **OpenAI Chat Completions** — direct via SGLang
- **OpenAI Responses** — via Gateway
- **Anthropic Messages** — via Gateway

## Compatibility matrix

| Client | API | Endpoint | Mode | Status | Notes |
|--------|-----|----------|------|--------|-------|
| OpenAI SDK | Chat Completions | `/v1/chat/completions` | Direct | ✅ Tested | Standard OpenAI format |
| OpenClaw | OpenAI-compatible | `/v1/chat/completions` | Direct | ✅ Tested | Direct SGLang endpoint |
| Hermes Agent | OpenAI-compatible | `/v1/chat/completions` | Direct | ✅ Tested | Direct SGLang endpoint |
| Claude Code | Anthropic Messages | `/v1/messages` | Gateway | 🧪 Experimental | See integrations/claude-code/ |
| Codex | OpenAI Responses | `/v1/responses` | Gateway | 🧪 Experimental | See integrations/codex/ |

## Direct mode

Use direct mode when the client supports OpenAI Chat Completions.

```
client → SGLang /v1/chat/completions → LnaLang4U engine
```

**Base URL:** `http://<host>:9000/v1`
**Model:** `deepseek-v4-flash`

## Gateway mode

Use gateway mode when the client expects Anthropic Messages API or OpenAI Responses API.

```
client → LnaLang4U Gateway → SGLang /v1/chat/completions → LnaLang4U engine
```

**Base URL:** `http://<host>:9010`
**Gateway container:** `lnalang4u-gateway`

## Status labels

- **✅ Tested** — basic request works and has been verified
- **🧪 Experimental** — works with known limitations; not fully validated
- **📋 Planned** — not tested yet

## Client-specific guides

- [OpenClaw](../integrations/openclaw/README.md)
- [Hermes Agent](../integrations/hermes-agent/README.md)
- [Claude Code](../integrations/claude-code/README.md)
- [Codex](../integrations/codex/README.md)
