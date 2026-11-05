# ──────────────────────────────────────────────────────────────────────────────
# ML Platform — Makefile
# Usage: make <target>
# ──────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash
.DEFAULT_GOAL := help

PAIR      ?= USD/BRL
EPOCHS    ?= 50
START     ?=
END       ?=
OUTPUT    ?= rates_raw.csv
COMPOSE   := docker compose -f docker-compose.yml
MONITORING := $(COMPOSE) -f docker-compose.monitoring.yml
IMAGE_TAG ?= latest

.PHONY: help install dev-install lint format typecheck test test-cov \
        fetch-data train serve infra-up infra-down infra-logs \
        monitoring-up monitoring-down clean docker-build

## ── Development ──────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

install:  ## Install production dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

dev-install: install  ## Install dev + production dependencies
	pip install -r requirements-dev.txt
	pre-commit install

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format code with ruff
	ruff format src/ tests/

typecheck:  ## Run mypy type checks
	mypy src/ --ignore-missing-imports

test:  ## Run unit tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

## ── Data & Training ──────────────────────────────────────────────────────────

fetch-data:  ## Fetch raw exchange rate data  [START=2020-01-01 END=2025-12-31 OUTPUT=rates_raw.csv]
	python -m src.ingestion.fetch_rates \
		$(if $(START),--start $(START)) \
		$(if $(END),--end $(END)) \
		--output $(OUTPUT)

train:  ## Train the LSTM model  [PAIR=USD/BRL EPOCHS=50]
	python -m src.training.train \
		--pair "$(PAIR)" \
		--epochs $(EPOCHS)

## ── Docker Infrastructure ────────────────────────────────────────────────────

docker-build:  ## Build all Docker images
	docker build -f Dockerfile.training -t ml-platform-training:$(IMAGE_TAG) .
	docker build -f Dockerfile.serving  -t ml-platform-serving:$(IMAGE_TAG)  .

infra-up:  ## Start core services (MLflow, serving, nginx, postgres)
	$(COMPOSE) up -d --build

infra-down:  ## Stop core services
	$(COMPOSE) down

infra-logs:  ## Tail logs from core services
	$(COMPOSE) logs -f --tail=100

monitoring-up:  ## Start full stack with monitoring
	$(MONITORING) up -d

monitoring-down:  ## Stop full stack
	$(MONITORING) down

serve:  ## Run API locally (no Docker)
	uvicorn src.serving.main:app --reload --port 8000

## ── Utilities ────────────────────────────────────────────────────────────────

clean:  ## Remove cache files and build artefacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml

healthcheck:  ## Check running services health
	@echo "=== Serving API ===" && curl -sf http://localhost/health && echo ""
	@echo "=== MLflow ===" && curl -sf http://localhost/mlflow/health && echo ""

mlflow-ui:  ## Open MLflow UI in browser (Linux)
	xdg-open http://localhost/mlflow/ 2>/dev/null || open http://localhost/mlflow/

grafana-ui:  ## Open Grafana in browser (Linux)
	xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000
