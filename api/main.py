import logging
import time
from pathlib import Path
from typing import ClassVar

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "heart_disease_pipeline.joblib"
logger = logging.getLogger("uvicorn.error")
START_TIME = time.time()

API_METRICS = {
    "total_requests": 0,
    "health_requests": 0,
    "predict_requests": 0,
    "predictions_generated": 0,
    "validation_errors_422": 0,
}

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description=(
        "Educational classifier for the Cleveland Heart Disease dataset. "
        "Inputs must use numeric ranges and category codes observed in the "
        "training data. Category codes: sex 0/1, cp 1-4, fbs 0/1, "
        "restecg 0-2, exang 0/1, slope 1-3, ca 0-3, thal 3/6/7. "
        "Model outputs are not medical advice, a diagnosis, or a clinically "
        "validated risk estimate."
    ),
)


@app.middleware("http")
async def log_api_requests(request: Request, call_next) -> Response:
    response = await call_next(request)
    path = request.url.path
    if path in {"/health", "/predict", "/metrics"}:
        API_METRICS["total_requests"] += 1
        if path == "/health":
            API_METRICS["health_requests"] += 1
        elif path == "/predict":
            API_METRICS["predict_requests"] += 1
            if response.status_code == 422:
                API_METRICS["validation_errors_422"] += 1
        logger.info(
            "API request completed: method=%s path=%s status_code=%s",
            request.method,
            path,
            response.status_code,
        )
    return response


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class PatientRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
                "thal": 3,
            }
        },
    )

    CATEGORY_CODES: ClassVar[dict[str, frozenset[float]]] = {
        "sex": frozenset({0, 1}),
        "cp": frozenset({1, 2, 3, 4}),
        "fbs": frozenset({0, 1}),
        "restecg": frozenset({0, 1, 2}),
        "exang": frozenset({0, 1}),
        "slope": frozenset({1, 2, 3}),
        "ca": frozenset({0, 1, 2, 3}),
        "thal": frozenset({3, 6, 7}),
    }

    age: float = Field(ge=29, le=77)
    sex: float
    cp: float
    trestbps: float = Field(ge=94, le=200)
    chol: float = Field(ge=126, le=564)
    fbs: float
    restecg: float
    thalach: float = Field(ge=71, le=202)
    exang: float
    oldpeak: float = Field(ge=0, le=6.2)
    slope: float
    ca: float
    thal: float

    @field_validator(
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    )
    @classmethod
    def validate_category_code(
        cls,
        value: float,
        info: ValidationInfo,
    ) -> float:
        field_name = info.field_name
        if field_name is None:
            raise ValueError(
                "Could not determine which category code to validate."
            )

        allowed_values = cls.CATEGORY_CODES[field_name]
        if value not in allowed_values:
            raise ValueError(
                f"{field_name} must be one of {sorted(allowed_values)}."
            )
        return value


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not found. Please run `python train.py` before "
                "starting the API."
            ),
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    load_model()
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.get("/metrics")
def metrics():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        **API_METRICS,
    }


@app.post("/predict")
def predict(record: PatientRecord):
    model = load_model()
    data = pd.DataFrame([record.model_dump()], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0, 1])
    API_METRICS["predictions_generated"] += 1
    logger.info(
        "Prediction generated for endpoint=/predict; input_values_logged=false"
    )
    label = (
        "Heart disease predicted"
        if prediction == 1
        else "No heart disease predicted"
    )
    return {
        "prediction": prediction,
        "probability": probability,
        "label": label,
    }