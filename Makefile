# ═══════════════════════════════════════════════════════════════════
# Makefile — Developer Workflow
# ═══════════════════════════════════════════════════════════════════
# Usage:
#   make help          — show all commands
#   make dev           — start full local stack
#   make test          — run all tests
#   make lint          — run linters
#   make build         — build Docker image
# ═══════════════════════════════════════════════════════════════════

.PHONY: help dev down build test test-unit test-integration lint fmt typecheck \
        security-scan k8s-apply-dev k8s-apply-qa k8s-apply-prod \
        k8s-rollback-qa k8s-rollback-prod logs clean

SHELL := /bin/bash
IMAGE_NAME ?= rag-langchain/api
IMAGE_TAG  ?= dev
NAMESPACE_PROD := rag-prod
NAMESPACE_QA   := rag-qa
NAMESPACE_DEV  := rag-dev

# ── Colours ──────────────────────────────────────────────────────────────
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RESET  := \033[0m

help: ## Show this help message
	@echo ""
	@echo "  $(CYAN)RAG LangChain — Developer Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Local Development ────────────────────────────────────────────────────

dev: ## Start full local stack (API + Qdrant + Ollama + Redis + Langfuse)
	@echo "$(CYAN)Starting local RAG stack...$(RESET)"
	@test -f .env || (cp .env.example .env && echo "$(YELLOW)Created .env from .env.example — edit before running$(RESET)")
	docker compose up -d
	@echo ""
	@echo "$(GREEN)Services started:$(RESET)"
	@echo "  API:         http://localhost:8000/docs"
	@echo "  Qdrant:      http://localhost:6333/dashboard"
	@echo "  Langfuse:    http://localhost:3000"
	@echo "  Ollama:      http://localhost:11434"
	@echo ""
	@echo "$(YELLOW)First run: Ollama is pulling llama3 — this takes a few minutes$(RESET)"

down: ## Stop all local services and remove containers
	docker compose down

down-clean: ## Stop all services AND remove volumes (resets all data)
	docker compose down -v
	@echo "$(YELLOW)All volumes deleted — fresh state$(RESET)"

logs: ## Follow API logs
	docker compose logs -f api

logs-all: ## Follow all service logs
	docker compose logs -f

restart-api: ## Rebuild and restart just the API container
	docker compose up -d --build api

# ── Build ────────────────────────────────────────────────────────────────

build: ## Build the production Docker image
	@echo "$(CYAN)Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)...$(RESET)"
	docker build \
		--target runtime \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--tag $(IMAGE_NAME):latest \
		--build-arg BUILD_DATE=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ") \
		.
	@echo "$(GREEN)Image built: $(IMAGE_NAME):$(IMAGE_TAG)$(RESET)"

build-multiplatform: ## Build for linux/amd64 + linux/arm64 (requires buildx)
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--target runtime \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--push \
		.

# ── Testing ──────────────────────────────────────────────────────────────

install-dev: ## Install all development dependencies
	pip install -e ".[dev]"

test: test-unit ## Run all tests (unit + integration if services available)
	@echo "$(GREEN)All tests passed$(RESET)"

test-unit: ## Run unit tests only (fast, no external services)
	@echo "$(CYAN)Running unit tests...$(RESET)"
	pytest tests/unit/ \
		-m unit \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v \
		-n auto
	@echo "$(GREEN)Unit tests passed. Coverage report: htmlcov/index.html$(RESET)"

test-integration: ## Run integration tests (requires docker compose up)
	@echo "$(CYAN)Running integration tests...$(RESET)"
	pytest tests/integration/ \
		-m integration \
		-v \
		--timeout=60
	@echo "$(GREEN)Integration tests passed$(RESET)"

test-watch: ## Run unit tests in watch mode (re-run on file change)
	ptw tests/unit/ -- -m unit -v

# ── Code Quality ─────────────────────────────────────────────────────────

lint: ## Run ruff linter
	@echo "$(CYAN)Linting...$(RESET)"
	ruff check . --fix
	@echo "$(GREEN)Lint passed$(RESET)"

fmt: ## Format code with ruff
	@echo "$(CYAN)Formatting...$(RESET)"
	ruff format .
	@echo "$(GREEN)Formatting done$(RESET)"

typecheck: ## Run mypy type checker
	@echo "$(CYAN)Type checking...$(RESET)"
	mypy app/ --ignore-missing-imports
	@echo "$(GREEN)Type check passed$(RESET)"

security-scan: ## Run bandit security scan + trivy (if docker image exists)
	@echo "$(CYAN)Security scanning...$(RESET)"
	bandit -r app/ -ll -x tests/ -f text
	@echo "$(GREEN)Bandit scan complete$(RESET)"
	@if docker image inspect $(IMAGE_NAME):$(IMAGE_TAG) > /dev/null 2>&1; then \
		echo "$(CYAN)Running Trivy CVE scan...$(RESET)"; \
		trivy image $(IMAGE_NAME):$(IMAGE_TAG) --severity HIGH,CRITICAL; \
	else \
		echo "$(YELLOW)No image found for Trivy scan. Run 'make build' first.$(RESET)"; \
	fi

quality: lint typecheck security-scan ## Run all quality checks

# ── Kubernetes ───────────────────────────────────────────────────────────

k8s-apply-dev: ## Apply all K8s manifests to dev namespace
	@echo "$(CYAN)Applying K8s manifests to $(NAMESPACE_DEV)...$(RESET)"
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/serviceaccount.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/qdrant-statefulset.yaml
	kubectl apply -f k8s/deployment.yaml
	@echo "$(GREEN)Dev manifests applied$(RESET)"

k8s-apply-qa: ## Apply all K8s manifests to QA namespace
	@echo "$(CYAN)Applying K8s manifests to $(NAMESPACE_QA)...$(RESET)"
	kubectl apply -f k8s/ -n $(NAMESPACE_QA)
	kubectl rollout status deployment/rag-api -n $(NAMESPACE_QA) --timeout=300s
	@echo "$(GREEN)QA deploy complete$(RESET)"

k8s-apply-prod: ## Apply all K8s manifests to prod namespace (WARNING: production)
	@echo "$(YELLOW)⚠️  Deploying to PRODUCTION namespace $(NAMESPACE_PROD)$(RESET)"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || exit 1
	kubectl apply -f k8s/ -n $(NAMESPACE_PROD)
	kubectl rollout status deployment/rag-api -n $(NAMESPACE_PROD) --timeout=600s
	@echo "$(GREEN)Production deploy complete$(RESET)"

k8s-status: ## Show status of all pods in all RAG namespaces
	@echo "$(CYAN)=== DEV ===$(RESET)"
	kubectl get pods,svc,hpa -n $(NAMESPACE_DEV) 2>/dev/null || echo "namespace not found"
	@echo "$(CYAN)=== QA ===$(RESET)"
	kubectl get pods,svc,hpa -n $(NAMESPACE_QA) 2>/dev/null || echo "namespace not found"
	@echo "$(CYAN)=== PROD ===$(RESET)"
	kubectl get pods,svc,hpa -n $(NAMESPACE_PROD) 2>/dev/null || echo "namespace not found"

k8s-rollback-qa: ## Roll back QA deployment to previous revision
	@echo "$(YELLOW)Rolling back QA...$(RESET)"
	kubectl rollout undo deployment/rag-api -n $(NAMESPACE_QA)
	kubectl rollout status deployment/rag-api -n $(NAMESPACE_QA) --timeout=300s
	@echo "$(GREEN)QA rollback complete$(RESET)"

k8s-rollback-prod: ## Roll back PRODUCTION deployment to previous revision
	@echo "$(YELLOW)⚠️  Rolling back PRODUCTION...$(RESET)"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || exit 1
	kubectl rollout undo deployment/rag-api -n $(NAMESPACE_PROD)
	kubectl rollout status deployment/rag-api -n $(NAMESPACE_PROD) --timeout=300s
	@echo "$(GREEN)Production rollback complete$(RESET)"

k8s-history: ## Show rollout history for prod deployment
	kubectl rollout history deployment/rag-api -n $(NAMESPACE_PROD)

k8s-scale-prod: ## Scale prod API to N replicas (usage: make k8s-scale-prod REPLICAS=5)
	kubectl scale deployment/rag-api --replicas=$(REPLICAS) -n $(NAMESPACE_PROD)

# ── Utilities ────────────────────────────────────────────────────────────

pull-models: ## Pull required Ollama models (run once after 'make dev')
	docker compose exec ollama ollama pull llama3
	docker compose exec ollama ollama pull nomic-embed-text
	docker compose exec ollama ollama list

ingest-sample: ## Ingest a sample PDF for testing (requires running API)
	@test -f sample.pdf || echo "No sample.pdf found. Download one first."
	curl -X POST http://localhost:8000/api/v1/ingest/file \
		-F "file=@sample.pdf" \
		-H "Accept: application/json"

query-sample: ## Run a sample query against the API
	curl -X POST http://localhost:8000/api/v1/query \
		-H "Content-Type: application/json" \
		-d '{"question": "What is the main topic of the ingested document?"}' \
		| python3 -m json.tool

clean: ## Remove generated files (.pyc, cache, coverage reports)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ coverage.xml .coverage

.DEFAULT_GOAL := help
