#!/usr/bin/env bash
# Claude Code smoke test via LnaLang4U Gateway
set -euo pipefail
BASE="${1:-http://127.0.0.1:9010}"
HEADERS=("-H" "Content-Type: application/json" "-H" "x-api-key: test" "-H" "anthropic-version: 2023-06-01")
PASS=0
FAIL=0

check() {
  local name="$1" code="$2"
  if [ "$code" = "200" ]; then
    echo "  ✅ $name"
    PASS=$((PASS+1))
  else
    echo "  ❌ $name (HTTP $code)"
    FAIL=$((FAIL+1))
  fi
}

echo "Claude Code smoke test → $BASE"
echo ""

# /v1/models
CODE=$(curl -s -o /dev/null -w "%{http_code}" "${HEADERS[@]}" "$BASE/v1/models")
check "/v1/models" "$CODE"

# /v1/messages text-only
RESP=$(curl -s -w "\n%{http_code}" "${HEADERS[@]}" "$BASE/v1/messages" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say hello in one word"}],"max_tokens":10}')
CODE=$(echo "$RESP" | tail -1)
check "/v1/messages" "$CODE"

# /v1/messages/count_tokens
CODE=$(curl -s -o /dev/null -w "%{http_code}" "${HEADERS[@]}" "$BASE/v1/messages/count_tokens" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}')
check "/v1/messages/count_tokens" "$CODE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
