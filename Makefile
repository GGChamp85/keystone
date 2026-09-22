# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
# Makefile for Keystone

.PHONY: help build sandbox-images up down logs db-init db-migrate api-key test lint clean certs certs-ca

COMPOSE := docker compose
APP := keystone-app
KEYSTONE_HOST ?= keystone.local

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: sandbox-images ## Build all containers (compose services + sandbox runtime images)
	$(COMPOSE) build

sandbox-images: ## Build sandbox runtime images (required before any Keystone Agents task can run — not a compose service, built standalone and referenced by name from src/sandbox/daemon.py's RUNTIME_IMAGES)
	docker build -f docker/sandbox-runtimes/python.Dockerfile -t keystone-sandbox-python:latest .

up: ## Start full stack (detached)
	$(COMPOSE) up -d

up-dev: ## Start only infra (postgres, redis, qdrant) for local dev
	$(COMPOSE) up -d postgres redis qdrant

up-demo: ## Start infra + the no-GPU demo model (llama.cpp, Qwen2.5-Coder-0.5B) — set VLLM_CODING_URL=http://demo-model:8000/v1
	$(COMPOSE) --profile demo-model up -d postgres redis qdrant sandbox-daemon demo-model
	@echo "Waiting for the demo model to load (first run downloads ~676 MB)..."
	@until curl -sf http://localhost:8090/health >/dev/null; do sleep 3; done
	@echo "demo-model ready at http://localhost:8090/v1 (inside compose: http://demo-model:8000/v1)"

down: ## Stop all containers
	$(COMPOSE) down

logs: ## Tail application logs
	$(COMPOSE) logs -f app

logs-vllm: ## Tail vLLM logs
	$(COMPOSE) logs -f vllm-coding vllm-reasoning vllm-codestral

db-init: ## Initialize database tables
	$(COMPOSE) exec app python scripts/setup_db.py

db-migrate: ## Run Alembic migrations
	$(COMPOSE) exec app alembic upgrade head

api-key: ## Create an API key (interactive)
	$(COMPOSE) exec app python scripts/create_api_key.py

test: ## Run tests
	python -m pytest tests/ -v --tb=short

lint: ## Lint with ruff, then type-check with mypy (same checks CI runs)
	ruff check src/ tests/ --fix
	ruff format src/ tests/
	mypy

health: ## Check platform health
	curl -s http://localhost:8080/health | python -m json.tool

status: ## Show container status
	$(COMPOSE) ps

clean: ## Remove all containers, volumes, and cached data
	$(COMPOSE) down -v --remove-orphans
	docker system prune -f

vllm-status: ## Check vLLM model health
	@echo "=== Coding (Qwen 32B) ===" && curl -s http://localhost:8001/v1/models | python -m json.tool 2>/dev/null || echo "DOWN"
	@echo "=== Reasoning (DeepSeek-R1) ===" && curl -s http://localhost:8002/v1/models | python -m json.tool 2>/dev/null || echo "DOWN"
	@echo "=== Codestral (22B) ===" && curl -s http://localhost:8003/v1/models | python -m json.tool 2>/dev/null || echo "DOWN"

certs: ## Generate a bare self-signed cert for a quick local dev boot (no CA, browsers will warn)
	mkdir -p nginx/certs
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout nginx/certs/privkey.pem \
		-out nginx/certs/fullchain.pem \
		-subj "/CN=localhost/O=Keystone" \
		-addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

certs-ca: ## Generate a real internal CA + CA-signed cert (pki/generate_ca.sh + issue_cert.sh) — use this for anything beyond a laptop
	@[ -f pki/ca/ca.key ] || ./pki/generate_ca.sh pki/ca
	./pki/issue_cert.sh $(KEYSTONE_HOST) pki/ca pki/certs DNS:localhost IP:127.0.0.1
	mkdir -p nginx/certs
	cp pki/certs/*.fullchain.crt nginx/certs/fullchain.pem
	cp pki/certs/*.key nginx/certs/privkey.pem
	@echo
	@echo "Distribute pki/ca/ca.crt to trust these certs (curl --cacert pki/ca/ca.crt, or"
	@echo "your OS/browser trust store) — NOT self-signed, verified against a real CA."
