"""Train, evaluate, and track the Heart Disease classification models."""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"
SUMMARY_PATH = PROJECT_ROOT / "models" / "training_summary.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_PATH = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "heart_disease_classification"


def configure_mlflow() -> None:
    """Use the project-local MLflow store and experiment."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(MLFLOW_PATH.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def build_pipeline(model) -> Pipeline:
    """Create one leakage-safe preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Cross-validate, test, save, and track one candidate model."""
    print(f"\nEvaluating {name.replace('_', ' ').title()}.")

    pipeline = build_pipeline(model)
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    cv_auc_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cross_validation,
        scoring="roc_auc",
    )

    pipeline.fit(X_train, y_train)
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = pipeline.predict(X_test)

    metrics = {
        "cv_roc_auc_mean": float(cv_auc_scores.mean()),
        "cv_roc_auc_std": float(cv_auc_scores.std()),
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, test_probabilities)
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / f"{name}_pipeline.joblib"
    joblib.dump(pipeline, artifact_path)
    print(f"Saved fitted candidate pipeline to: {artifact_path}")

    figure, axis = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, test_probabilities, ax=axis)
    axis.set_title(f"ROC Curve - {name.replace('_', ' ').title()}")
    figure.tight_layout()
    roc_path = ARTIFACT_DIR / f"{name}_roc_curve.png"
    figure.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(artifact_path),
            artifact_path="model_artifact",
        )
        mlflow.log_artifact(
            str(roc_path),
            artifact_path="plots",
        )
        mlflow.log_param("model_name", name)

    print(f"Logged MLflow run: {run.info.run_id}")
    return pipeline, metrics, run.info.run_id


def create_candidate_models() -> dict:
    """Define the baseline classifiers and their fixed hyperparameters."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
        ),
    }


def save_final_model(
    best_run: dict,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict:
    """Refit the selected pipeline on all data and write model and summary."""
    final_pipeline = best_run["pipeline"]
    final_pipeline.fit(features, target)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    summary = {
        "best_model": best_run["name"],
        "model_path": str(MODEL_PATH),
        "metrics": {
            metric: best_run[metric]
            for metric in ("roc_auc", "accuracy", "precision", "recall")
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Run data loading, model comparison, tracking, and final saving."""
    print("Heart Disease model training and experiment tracking")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {DATA_PATH}. "
            "Run python download.py or the data-preparation notebook first."
        )

    print("\nStep 1/4: Loading the cleaned dataset.")
    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=["target"])
    target = data["target"]
    print(
        f"Loaded {len(data):,} patient records with "
        f"{len(features.columns)} predictors."
    )
    print("Target counts (0 = no disease, 1 = disease):")
    print(target.value_counts().sort_index().to_string())
    missing_predictors = features.isna().sum()
    missing_predictors = missing_predictors[missing_predictors > 0]
    if missing_predictors.empty:
        print("Missing predictor values: none.")
    else:
        print(
            "Missing predictor values "
            "(handled by pipeline median imputation):"
        )
        print(missing_predictors.to_string())

    print("\nStep 2/4: Creating the stratified train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    print(
        f"Training rows: {len(X_train):,} | "
        f"Test rows: {len(X_test):,} | "
        "Test fraction: 20%"
    )
    print(
        "Each candidate uses five-fold stratified cross-validation "
        "on the training split."
    )

    print("\nStep 3/4: Training and comparing the baseline classifiers.")
    configure_mlflow()
    candidate_models = create_candidate_models()
    results = []

    for name, model in candidate_models.items():
        pipeline, metrics, run_id = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        results.append(
            {
                "name": name,
                "run_id": run_id,
                **metrics,
                "pipeline": pipeline,
            }
        )

    results_table = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key != "pipeline"}
            for result in results
        ]
    ).set_index("name")
    print("\nModel comparison (test metrics use the held-out 20% split):")
    print(results_table.drop(columns="run_id").round(3).to_string())

    print("\nStep 4/4: Selecting and saving the final model.")
    print(
        "Selection rule: choose the model with the highest held-out "
        "test ROC-AUC. This assignment demonstration reuses the test "
        "split for selection, which can make the reported winner's "
        "metrics optimistic."
    )
    best_run = max(results, key=lambda result: result["roc_auc"])
    summary = save_final_model(best_run, features, target)

    print(f"Selected model: {best_run['name']}")
    print(f"Saved final pipeline to: {MODEL_PATH}")
    print(f"Saved training summary to: {SUMMARY_PATH}")
    print(f"MLflow tracking data: {MLFLOW_PATH}")
    print("\nFinal model summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
