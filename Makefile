.PHONY: up-engine up-gateway up-full down \
        smoke-openai smoke-anthropic smoke-all \
        build-engine build-gateway

# --- Build ---

build-engine:
	docker build -f sglang-diskkv/Dockerfile.sglang-dsv4 \
	  -t sglang-dsv4-diskkv:latest .

build-gateway:
	docker build -f gateway/Dockerfile -t lnalang4u-gateway:latest .

# --- Start ---

up-engine:
	docker compose -f compose/docker-compose.engine.yml up -d

up-gateway:
	docker compose -f compose/docker-compose.gateway.yml up -d

up-full:
	docker compose -f compose/docker-compose.full.yml up -d

# --- Stop ---

down:
	docker compose -f compose/docker-compose.full.yml down 2>/dev/null || true
	docker compose -f compose/docker-compose.engine.yml down 2>/dev/null || true
	docker compose -f compose/docker-compose.gateway.yml down 2>/dev/null || true

# --- Smoke tests ---

smoke-openai:
	python3 benchmarks/agent/scripts/run_openai_smoke.py

smoke-anthropic:
	python3 benchmarks/agent/scripts/run_anthropic_smoke.py

smoke-all: smoke-openai smoke-anthropic

# --- Help ---

help:
	@echo "Targets:"
	@echo "  build-engine    Build the SGLang + DiskOffload Docker image"
	@echo "  build-gateway   Build the API compatibility gateway"
	@echo "  up-engine       Start inference engine (requires MODEL_DIR, DSV4_KERNEL_DIR)"
	@echo "  up-gateway      Start gateway alone (point at running engine)"
	@echo "  up-full         Start full stack (engine + gateway)"
	@echo "  down            Stop all containers"
	@echo "  smoke-openai    Test OpenAI-compatible endpoint"
	@echo "  smoke-anthropic Test Anthropic-compatible endpoint"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL_DIR       Path to DeepSeek-V4-Flash-FP8 checkpoint"
	@echo "  DSV4_KERNEL_DIR Path to SM120 kernel build-docker"
	@echo "  DISKKV_DIR      Path to SSD KV cache directory"
	@echo "  CONTEXT_LENGTH  Context length (default: 32768)"
	@echo "  GPUS            GPU indices (default: 0,2,3,4)"
