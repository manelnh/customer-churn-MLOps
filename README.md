# Churn Prediction MLOps App

This project packages a churn prediction workflow with:

- a Streamlit application for production-style inference and review
- PostgreSQL for prediction logs, governance records, and monitoring alerts
- MLflow for experiment tracking and monitoring run history
- GitHub Actions pipelines for CI, container delivery, scheduled monitoring, and scheduled retraining
- Alembic migrations for reproducible database schema management

## Local Startup

The easiest first-time setup is Docker Compose:

```powershell
docker compose up --build
```

If you want to initialize the database schema explicitly before running the app:

```powershell
alembic upgrade head
```

Then open:

- Streamlit app: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

## Core User Flow

1. Open the `Single Prediction` tab and run a few customer predictions.
2. Review risk summaries in `Manager Insights`.
3. Save operational decisions in `Action Center`.
4. Use `Technical Lab` to compare a candidate model and log it to MLflow.
5. Open `Monitoring Dashboard` to review live evidence and stored alerts.

## Running Monitoring From The Terminal

The monitoring script can be run manually or on a scheduler without opening Streamlit:

```powershell
python Scripts/run_monitoring.py
```

Useful options:

```powershell
python Scripts/run_monitoring.py --days 14
python Scripts/run_monitoring.py --skip-mlflow-logging
python Scripts/run_monitoring.py --threshold 2 --fail-on-high-alerts
```

What it does:

- reads recent production predictions from PostgreSQL
- computes live coverage and drift-style alerts
- logs monitoring runs to MLflow unless skipped
- prints whether retraining should be considered

## Running Training From The Terminal

You can also trigger the end-to-end training pipeline directly:

```powershell
python Scripts/run_training.py --reason manual_validation
```

What it does:

- loads the churn dataset
- trains multiple logistic regression variants
- logs experiments to MLflow
- saves the best production bundle locally
- attempts MLflow model registration/promotion

## Tests

Run the lightweight unit suite with:

```powershell
python -m unittest discover -s tests -v
```

## Migrations

Alembic is included so the PostgreSQL schema can evolve in a reproducible way:

```powershell
alembic upgrade head
```

The initial migration creates:

- `prediction_logs`
- `monitoring_alerts`
- `governance_decisions`

## CI

GitHub Actions CI is defined in `.github/workflows/ci.yml` and now performs:

- dependency installation with pip cache
- Python lint checks for critical import/name errors
- bytecode compilation checks
- unit tests with coverage
- Docker image build validation

## CD And Automation

Additional GitHub Actions workflows are included:

- `.github/workflows/cd.yml`
  Builds and pushes a Docker image to GitHub Container Registry on `main`/`master` and tags. If `DEPLOY_WEBHOOK_URL` is configured as a GitHub secret, it also triggers deployment automatically.
- `.github/workflows/monitoring.yml`
  Runs a scheduled monitoring cycle every day and can also be launched manually.
- `.github/workflows/training.yml`
  Runs scheduled retraining every Monday and supports manual dispatch.

## Required GitHub Secrets

To activate the full MLOps automation on GitHub, configure these repository or environment secrets:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `DEPLOY_WEBHOOK_URL` for optional deployment triggering

## Project MLOps Coverage

This repository now covers the main blocks expected in a complete PFE-style MLOps project:

- data-driven model training and experiment tracking
- production inference and prediction logging
- governance decision logging
- monitoring and retraining signals
- schema migrations
- CI quality gates
- CD-ready container publishing
- scheduled operational automation
