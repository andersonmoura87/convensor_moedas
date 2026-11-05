# ML Platform — Exchange Rate Forecaster

> End-to-end MLOps platform for forecasting currency exchange rates using a multi-layer LSTM (PyTorch), with full DevOps infrastructure for a Linux production server.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Linux Server (Ubuntu 22.04)                  │
│                                                                     │
│  ┌─────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  Nginx  │──▶│  FastAPI │──▶│  LSTM Model  │──▶│   MLflow    │  │
│  │ (proxy) │   │ (serving)│   │  (PyTorch)   │   │  Registry   │  │
│  └─────────┘   └──────────┘   └──────────────┘   └─────────────┘  │
│                     │                                    │          │
│  ┌──────────────────▼────────────────────────────────────▼───────┐ │
│  │              Observability Stack                               │ │
│  │   Prometheus ──▶ Grafana ◀── Loki ◀── Promtail               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  CI/CD (GitHub Actions)                                      │  │
│  │  lint → test → docker build → security scan → SSH deploy     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ml-platform/
├── src/
│   ├── ingestion/      # Frankfurter API client (retries, rate limiting)
│   ├── features/       # Feature engineering (lags, rolling stats, log-return)
│   ├── training/       # LSTM model + MLflow training pipeline
│   └── serving/        # FastAPI app with Prometheus metrics
├── tests/              # Unit tests (70%+ coverage enforced in CI)
├── monitoring/
│   ├── prometheus/     # Scrape config
│   ├── grafana/        # Auto-provisioned datasources + dashboard
│   ├── loki/           # Log aggregation config
│   └── promtail/       # Docker log shipping
├── infra/nginx/        # Reverse proxy + rate limiting
├── scripts/            # setup_server.sh — bootstrap Ubuntu from zero
├── .github/workflows/  # ci.yml (lint/test/build) + cd.yml (SSH deploy)
├── docker-compose.yml           # Core services
├── docker-compose.monitoring.yml # Observability stack
├── Dockerfile.training
├── Dockerfile.serving
├── dvc.yaml            # Reproducible data + training pipeline
├── params.yaml         # Versioned hyperparameters
└── Makefile            # Developer ergonomics
```

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Docker + Docker Compose v2
- `make`

```bash
# 1. Clone and install
git clone <repo>
cd ml-platform
make dev-install

# 2. Fetch historical exchange rates (last 5 years)
make fetch-data

# 3. Start core infrastructure (MLflow, Postgres, Nginx)
make infra-up

# 4. Train the model (logs to MLflow)
make train PAIR=USD/BRL EPOCHS=50

# 5. Start full stack with monitoring
make monitoring-up

# 6. Check health
make healthcheck
```

**Access points:**

| Service | URL | Notes |
|---|---|---|
| API docs | `http://localhost/api/v1/docs` | Swagger UI |
| MLflow UI | `http://localhost/mlflow/` | Experiments & registry |
| Grafana | `http://localhost:3000` | Default: admin / grafana_secret |
| Prometheus | `http://localhost:9090` | Raw metrics |

## ML Pipeline

### Data
- Source: [Frankfurter API](https://www.frankfurter.app/) (European Central Bank data, free, no key)
- Pairs: USD/BRL, EUR/BRL, EUR/USD, GBP/USD, JPY/USD
- History: up to 25 years of daily rates

### Features (25 total)
| Category | Features |
|---|---|
| Lags | t-1, t-2, t-3, t-5, t-7, t-14, t-21 |
| Rolling | mean/std/min/max at 7, 14, 30 days |
| Calendar | day_of_week, month, quarter, is_month_end, year |
| Returns | log_return (first-difference of log price) |

### Model — LSTM Forecaster
```
Input (seq_len=30, n_features=25)
  → LSTM ×2 layers (hidden=128, dropout=0.2)
  → FC (128 → 64, ReLU, Dropout)
  → FC (64 → 1)
  → Scalar rate prediction
```

Training:
- Loss: Huber (robust to outliers)
- Optimiser: Adam + ReduceLROnPlateau
- Early stopping: patience=10
- Chronological 70/15/15 train-val-test split (no leakage)

### Metrics tracked (MLflow)
- `train_loss`, `val_loss` per epoch
- `val_mae`, `val_rmse`, `val_mape`
- `test_mae`, `test_rmse`, `test_mape`

## MLOps

### Experiment tracking (MLflow)
```bash
# All experiments, params, metrics and artefacts stored in Postgres + volume
mlflow ui  # or access via http://localhost/mlflow/
```

### Data versioning (DVC)
```bash
dvc repro          # Run full pipeline (fetch → features → train)
dvc params diff    # Show hyperparameter changes vs last run
dvc metrics show   # Show tracked metrics
```

### Model Registry
The training script automatically registers the best model under `exchange-rate-lstm`.
Promote to Production via MLflow UI or:
```python
from mlflow import MlflowClient
client = MlflowClient()
client.transition_model_version_stage("exchange-rate-lstm", version=1, stage="Production")
```

## API Reference

### `POST /predict`
```json
{
  "pair": "USD/BRL",
  "features": [[...25 values...], ...]  // shape: (30, 25)
}
```
Response:
```json
{
  "pair": "USD/BRL",
  "predicted_rate": 5.12,
  "latency_ms": 8.3
}
```

### `GET /metrics`
Prometheus text format — scraped every 15s by Prometheus.

## DevOps

### CI Pipeline (GitHub Actions)
```
push → lint (ruff + mypy) → tests (pytest, 70% coverage) → docker build → security audit
```

### CD Pipeline
```
merge to main → SSH into server → docker compose pull → rolling restart → health check
```

### Production server setup
```bash
# On a fresh Ubuntu 22.04 server:
curl -fsSL https://raw.githubusercontent.com/.../scripts/setup_server.sh | sudo bash
```

The script installs Docker, configures UFW firewall, fail2ban, and registers a systemd service.

### Required GitHub Secrets
| Secret | Description |
|---|---|
| `DEPLOY_SSH_KEY` | Private SSH key for server access |
| `SERVER_HOST` | Server IP or hostname |
| `SERVER_USER` | SSH user (e.g. `mlplatform`) |

## Observability

### Dashboards (auto-provisioned in Grafana)
- **Prediction Requests/min** — by pair and status
- **P95 Latency** — sliding 5-minute window
- **Model Loaded** — binary gauge (GREEN = ready)
- **Error Rate** — with color thresholds at 1% and 5%
- **Application Logs** — live tail from Loki

### Alerts (extend in `monitoring/prometheus/rules/`)
```yaml
- alert: ModelNotLoaded
  expr: model_loaded == 0
  for: 2m
- alert: HighErrorRate
  expr: rate(prediction_requests_total{status="error"}[5m]) / rate(prediction_requests_total[5m]) > 0.05
```

## Development

```bash
make lint       # ruff check
make format     # ruff format
make typecheck  # mypy
make test       # pytest
make test-cov   # pytest + HTML coverage report
make clean      # remove all cache/build artefacts
```

## License

MIT
