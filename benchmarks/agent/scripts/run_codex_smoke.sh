#!/usr/bin/env bash
# Codex-compatible Responses API smoke test
set -euo pipefail
BASE="${1:-http://127.0.0.1:9010}"
echo "Codex smoke test → $BASE"
echo ""

echo -n "/v1/responses (text input)... "
R=$(curl -s -w "\n%{http_code}" "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"codex-lnalang4u","input":"Say OK","max_output_tokens":10}')
CODE=$(echo "$R" | tail -1)
BODY=$(echo "$R" | sed '$d')
if [ "$CODE" = "200" ]; then
  OUT=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('output',[{}])[0].get('content',[{}])[0].get('text','')[:20])" 2>/dev/null)
  echo "✅ $CODE — $OUT"
else
  echo "❌ $CODE"
fi

echo -n "/v1/responses (message input)... "
R=$(curl -s -w "\n%{http_code}" "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"codex-lnalang4u","input":[{"role":"user","content":"Say OK"}],"max_output_tokens":10}')
CODE=$(echo "$R" | tail -1)
echo "$([ "$CODE" = "200" ] && echo "✅" || echo "❌") $CODE"
