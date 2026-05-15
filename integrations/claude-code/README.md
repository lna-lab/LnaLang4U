# Claude Code integration

Claude Code support is provided through the LnaLang4U Gateway. The gateway exposes an Anthropic Messages-compatible surface:

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /v1/models`

## Environment

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9010
export ANTHROPIC_API_KEY=local-dev
```

Then run Claude Code normally:

```bash
claude
```

## Status

🧪 Experimental until the following are tested:

- [ ] text-only request
- [ ] streaming response
- [ ] tool use
- [ ] large context prompt
- [ ] code edit task
