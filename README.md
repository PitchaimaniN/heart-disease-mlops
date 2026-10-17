# Heart Disease MLOps Project

An end-to-end machine-learning project using the Cleveland Heart Disease dataset. It includes data preparation, exploratory data analysis, model training and MLflow tracking, a FastAPI prediction service with a monitoring metrics endpoint, Docker packaging, Kubernetes manifests, and a GitHub Actions workflow.

## Project contents

- `download.py` downloads the UCI Cleveland source data and creates the cleaned dataset.
- `train.py` compares Logistic Regression and Random Forest, logs both runs to MLflow, and saves the final preprocessing-and-model pipeline along with ROC curve plots.
- `notebooks/` contains interactive notebooks for data preparation, EDA, training/tracking, and inference.
- `api/main.py` provides `/health`, `/metrics`, and `/predict` endpoints.
- `tests/` contains unit tests for data processing, model pipeline imputation, and API endpoints.
- `.github/workflows/main.yml` automates data download, training, linting, and tests.
- `Dockerfile` builds the API container; `deployment.yaml` and `service.yaml` define its Kubernetes Deployment and Service separately.

## Setup

Run these commands in PowerShell from the project root:

```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` creates the project `.venv` if needed, installs `requirements.txt`, and registers the `Python (heart-disease-mlops)` Jupyter kernel. In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter and select `Python (heart-disease-mlops)` as the notebook kernel; both should use the project's `.venv`.

## Prepare the data

Choose one method; the notebook and script perform the same download-and-clean operation.

```powershell
python download.py
```

Or run `notebooks/01_data_acquisition_and_cleaning.ipynb` in VS Code. Either option downloads the UCI source and overwrites:

- `data/raw/heart.csv`
- `data/processed/heart_clean.csv`

Internet access is required. The cleaned dataset has 303 records, 13 predictors, and a binary `target` (0 = no heart disease, 1 = heart disease). In this source dataset, four `ca` and two `thal` predictor values are missing; the model pipeline imputes them during training.

## Exploratory data analysis

Run all cells in `notebooks/eda.ipynb`. It summarizes the dataset, reports missingness and target counts, and creates labeled feature histograms, a correlation heatmap, and a class-balance chart. The figures are saved under `report/figures/`.

## Train and track models

Choose either the notebook or script:

```powershell
python train.py
```

Or run `notebooks/02_model_training_and_tracking.ipynb` from top to bottom. Both workflows compare baseline Logistic Regression and Random Forest pipelines, using median imputation, scaling, a stratified 80/20 split, and five-fold stratified cross-validation on the training split. They report accuracy, precision, recall, and ROC-AUC, and log model parameters, metrics, artifacts, and ROC plots to MLflow.

The selected pipeline is saved to `models/heart_disease_pipeline.joblib`; the selection summary is saved to `models/training_summary.json`. 

### View MLflow runs

After training, start the UI from the project root in a separate PowerShell terminal:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

Keep that terminal open and visit `http://127.0.0.1:5000`. Runs are stored locally in `mlruns/` and are excluded from Git by `.gitignore`.

## Run local model inference

Run `notebooks/03_model_inference.ipynb` after training. It loads the saved pipeline and makes a local example prediction. Its optional API-request cell requires the API to be running as described below.

Predictions and probabilities are educational model outputs, not medical advice, diagnoses, or clinically validated risk estimates.

## Run and test the API

Start the service from the project root after training:

```powershell
python -m uvicorn api.main:app --reload --no-access-log --host 127.0.0.1 --port 8000
```

Keep the server terminal open. Browse to:

- `http://127.0.0.1:8000/health` to check that the service can load the model.
- `http://127.0.0.1:8000/metrics` to view real-time API performance and request counters.
- `http://127.0.0.1:8000/docs` for Swagger UI, where you can try endpoints and submit sample requests.

The prediction request requires 13 numeric predictor fields and must not include `target`. Use the categories and ranges represented in the prepared dataset (for example, `cp` is coded 1-4):

```json
{
  "age": 52,
  "sex": 1,
  "cp": 2,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

A successful response includes `prediction` (0 or 1), `probability` (the model's estimated probability for class 1), and a human-readable `label`. Inputs outside the observed training-data ranges or category codes, and unexpected fields such as `target`, are rejected with HTTP 422. 

For automated unit tests and linting, open another project-root terminal and run:

```powershell
pytest -q tests
flake8 download.py train.py api tests
```

The API logs request method, route, response status, and a generic prediction-generated event. It does not log request fields, patient values, prediction class, probability, or caller identity. Uvicorn access logs are disabled in the documented local and container commands so they do not add client IP addresses.

## Docker

Train the model first, then build and run the container image:

```powershell
docker build -t heart-disease-mlops .
docker run --rm --name heart-disease-api -p 8000:8000 heart-disease-mlops
```

With the container running, check `http://127.0.0.1:8000/health` or `http://127.0.0.1:8000/metrics`. Stop the container with `Ctrl+C`.

## Kubernetes

Build the trained API image, load/import it into your local cluster node (e.g., Docker Desktop or Minikube), and apply the manifests:

```powershell
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get deployments,pods,services
```

Access the service locally via port-forward:
```powershell
kubectl port-forward service/heart-disease-api-service 8001:80
```

## GitHub Actions

The workflow runs on pushes to `main` or `master` and on pull requests. It downloads data, trains models, runs Flake8, and executes pytest (all 11 tests). It uploads the final model, training summary, candidate pipelines, MLflow run data, and ROC plots as workflow artifacts retained for 14 days.