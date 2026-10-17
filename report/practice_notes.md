# Heart Disease MLOps Project - Practice and Review Notes

Use these notes to rehearse the workflow, inspect what each stage produces, and explain the design choices in your own words. The goal is to understand the project rather than memorize commands.

## The pipeline in one sentence

**Source data -> clean -> explore -> train and track -> serve -> package -> deploy -> automate -> show evidence**

Each stage produces an input or proof needed by a later stage:

| Stage | Main output | Used by |
| --- | --- | --- |
| Data preparation | `data/processed/heart_clean.csv` | EDA and training |
| EDA | Charts under `report/figures/` | Analysis and report |
| Training | Saved model pipeline and MLflow runs | API and Docker |
| API | Health and prediction endpoints | Local, Docker, and Kubernetes checks |
| Docker | Container image | Local container and Kubernetes |
| Kubernetes | Running API pods and Service | Deployment verification |
| GitHub Actions | Automated checks and workflow artifacts | Continuous integration evidence |

## Practice the workflow

For every stage, try this pattern:

1. Say what the stage is for before running it.
2. Run the command or notebook cells.
3. Inspect the output and identify what was created.
4. Explain how the next stage uses that output.

### 1. Prepare the environment

From the project root, follow the setup instructions in the README. Confirm that the terminal is using the project's `.venv` and that required packages are installed.

**Explain in your own words:** Why is it useful for a project to declare its dependencies and use a consistent Python environment?

### 2. Download and clean the data

Run:

```powershell
python download.py
```

The script downloads the Cleveland source data, reads `?` as a missing value, converts the original outcome to a binary target, and saves the prepared CSV. The target mapping is `0 = no heart disease`; original outcome values `1` through `4` become `1 = heart disease`.

Inspect `data/processed/heart_clean.csv` and the script output. Note the row and column counts, target counts, and missing predictor values.

**Explain in your own words:**

- What does target `0` mean? What does target `1` mean?
- Why are missing predictor values retained for imputation instead of dropping every affected row?
- Why is it important to keep the target column separate from the predictors?

### 3. Explore the data

Run all cells in `notebooks/eda.ipynb`. Inspect the feature histograms, correlation heatmap, and target class-balance chart saved under `report/figures/`.

**Explain in your own words:**

- What can a histogram show about the distribution and range of a feature?
- What does a correlation heatmap summarize, and why does correlation not establish causation?
- Why is the target class balance useful when choosing how to split the data?
- Why should patterns from this small historical dataset not automatically be generalized to current patients?

### 4. Train and compare models

Run:

```powershell
python train.py
```

The script compares Logistic Regression and Random Forest. Their preprocessing and estimator are combined in pipelines. The code uses median imputation, scaling, a stratified train/test split, and five-fold stratified cross-validation on the training split. It records experiment parameters, metrics, and artifacts in MLflow, and saves a final model pipeline.

If desired, start the MLflow UI using the README instructions and compare the recorded runs.

**Explain in your own words:**

- Why put imputation, scaling, and the estimator in one pipeline?
- Why stratify a train/test split?
- How are cross-validation and the held-out test split different?
- What do accuracy, precision, recall, and ROC-AUC describe?
- Why might selecting a model using the held-out test ROC-AUC make the reported test results optimistic?

### 5. Check the API and tests

Run the API tests:

```powershell
pytest -q tests
```

Start the API as shown in the README, then open `http://127.0.0.1:8000/docs`. Try `GET /health` and a valid `POST /predict` using the example request in the README.

The health endpoint checks that the service can load its model. The prediction endpoint validates the input, creates a prediction, and returns a class, probability, and label. Invalid feature values and unexpected fields such as `target` are rejected.

**Explain in your own words:**

- What does a successful health check establish, and what does it not establish?
- Why must a prediction request not include the answer (`target`) as an input?
- Why reject unknown fields and category codes outside the supported set?
- Which details are deliberately excluded from application logs, and why?

### 6. Build and run Docker

After training has created the model file, follow the Docker build and run commands in the README.

**Explain in your own words:**

- What does the Docker image package?
- Why must the trained model exist before building the image?
- How is running the API in a container different from running it directly in the Python environment?

### 7. Deploy to Kubernetes

Follow the Kubernetes section in the README. Apply the project's `deployment.yaml` and `service.yaml` manifests, then check the deployment, pods, Service, and application logs. If Kubernetes reports `ErrImageNeverPull`, follow the Docker Desktop image-import steps in the README.

A Deployment manages the desired number of API pods. A Service provides a stable way to reach ready pods. In Docker Desktop, the Kubernetes node may use a separate containerd image store, so it may need the locally built image to be explicitly imported.

**Explain in your own words:**

- What does the Deployment manage?
- What problem does the Service solve?
- Why might Docker show an image that Kubernetes still cannot find?
- What evidence shows that the deployment is ready and responding?

### 8. Review continuous integration

Open `.github/workflows/main.yml` and the successful GitHub Actions run. The workflow downloads data, trains models, runs linting and tests, and uploads training artifacts.

**Explain in your own words:**

- Which steps run automatically in GitHub Actions?
- What does a green workflow run demonstrate?
- Why are uploaded artifacts useful when the workflow runs on GitHub-hosted machines?

## Two-minute video rehearsal

Prepare the windows and ensure the API or Kubernetes deployment is ready before starting the recording.

| Time | Show and explain |
| --- | --- |
| 0:00-0:15 | Introduce the project and summarize the end-to-end pipeline. |
| 0:15-0:30 | Show the EDA notebook and its histograms, correlation heatmap, and class-balance chart. |
| 0:30-0:45 | Show the model comparison and MLflow runs. |
| 0:45-1:05 | Show the test result and the API health/prediction checks in Swagger. |
| 1:05-1:25 | Show the Docker image or running container. |
| 1:25-1:45 | Show Kubernetes pods and Service, plus sanitized application logs if available. |
| 1:45-2:00 | Show the successful GitHub Actions run and summarize what it verifies. |

Explain the steps in your own words. Use only the project example data; do not show secrets or real patient information. Save the recording as `report/evidence/end_to_end_pipeline.mp4`.

## Self-check

Try answering these without looking at the notes. If an answer is unclear, return to that project stage, inspect the code or output, and explain it again in your own words.

1. Which files does data preparation create, and how is the target mapped?
2. Why are missing predictor values imputed inside the model pipeline?
3. How does cross-validation differ from a held-out test split?
4. What do the API validation rules prevent, and what information do the logs avoid recording?
5. How do Docker and Kubernetes play different roles in this project?
6. Which checks does GitHub Actions run automatically?
7. What are two important limitations of this dataset or evaluation?

## Suggested review order

Use the README as the command guide. For implementation details, inspect `download.py`, `train.py`, `api/main.py`, `tests/`, `Dockerfile`, `deployment.yaml`, `service.yaml`, and `.github/workflows/main.yml`. Use the final report to review the results, evidence, and limitations.
