import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------
# Pydantic Schemas with Robust Validation & Clean Examples
# ---------------------------------------------------------

class IrisFeatures(BaseModel):
    """
    Input schema representing the 4 morphological features of an Iris flower.
    All measurements must be positive numbers in centimeters.
    """
    sepal_length: float = Field(
        ...,
        gt=0.0,
        le=15.0,
        description="Length of the sepal in centimeters (cm)",
        examples=[5.1]
    )
    sepal_width: float = Field(
        ...,
        gt=0.0,
        le=15.0,
        description="Width of the sepal in centimeters (cm)",
        examples=[3.5]
    )
    petal_length: float = Field(
        ...,
        gt=0.0,
        le=15.0,
        description="Length of the petal in centimeters (cm)",
        examples=[1.4]
    )
    petal_width: float = Field(
        ...,
        gt=0.0,
        le=15.0,
        description="Width of the petal in centimeters (cm)",
        examples=[0.2]
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_envelope(cls, data: Any) -> Any:
        """
        Gracefully handles OpenAPI example envelopes if submitted directly:
        e.g., {'summary': '...', 'description': '...', 'value': {...}}
        """
        if isinstance(data, dict) and "value" in data and isinstance(data["value"], dict):
            return data["value"]
        return data

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                {
                    "sepal_length": 6.0,
                    "sepal_width": 2.9,
                    "petal_length": 4.5,
                    "petal_width": 1.5
                },
                {
                    "sepal_length": 6.7,
                    "sepal_width": 3.0,
                    "petal_length": 5.2,
                    "petal_width": 2.3
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """
    Output schema for a single prediction request.
    """
    predicted_class_id: int = Field(
        ...,
        description="Integer class index (0 = setosa, 1 = versicolor, 2 = virginica)",
        examples=[0]
    )
    predicted_class_name: str = Field(
        ...,
        description="Human-readable botanical species name",
        examples=["setosa"]
    )
    confidence: float = Field(
        ...,
        description="Confidence score / probability for the predicted species",
        examples=[0.985]
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Predicted probability distribution across all Iris classes",
        examples=[{"setosa": 0.985, "versicolor": 0.010, "virginica": 0.005}]
    )
    input_features: IrisFeatures = Field(
        ...,
        description="Echo of the input features received by the model"
    )


class BatchIrisFeatures(BaseModel):
    """
    Input schema for running inference on multiple Iris flower samples at once.
    """
    samples: List[IrisFeatures] = Field(
        ...,
        min_length=1,
        description="List of Iris sample feature sets",
        examples=[
            [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
                {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
            ]
        ]
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_batch_envelope(cls, data: Any) -> Any:
        if isinstance(data, dict) and "value" in data and isinstance(data["value"], dict):
            return data["value"]
        return data


class BatchPredictionResponse(BaseModel):
    """
    Output schema for batch prediction requests.
    """
    total_samples: int = Field(..., description="Number of samples evaluated", examples=[3])
    predictions: List[PredictionResponse] = Field(..., description="List of individual predictions")


class ModelInfoResponse(BaseModel):
    """
    Schema describing metadata about the loaded machine learning model.
    """
    model_name: str
    feature_names: List[str]
    target_names: List[str]
    description: str
    status: str


# ---------------------------------------------------------
# OpenAPI Swagger Dropdown Examples
# ---------------------------------------------------------

PREDICT_EXAMPLES = {
    "setosa": {
        "summary": "Iris Setosa Sample",
        "description": "Typical measurements for an Iris Setosa flower",
        "value": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    },
    "versicolor": {
        "summary": "Iris Versicolor Sample",
        "description": "Typical measurements for an Iris Versicolor flower",
        "value": {
            "sepal_length": 6.0,
            "sepal_width": 2.9,
            "petal_length": 4.5,
            "petal_width": 1.5
        }
    },
    "virginica": {
        "summary": "Iris Virginica Sample",
        "description": "Typical measurements for an Iris Virginica flower",
        "value": {
            "sepal_length": 6.7,
            "sepal_width": 3.0,
            "petal_length": 5.2,
            "petal_width": 2.3
        }
    }
}

BATCH_PREDICT_EXAMPLES = {
    "three_species": {
        "summary": "Batch of 3 Species",
        "description": "Contains one sample from each Iris species",
        "value": {
            "samples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
                {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
            ]
        }
    }
}


# ---------------------------------------------------------
# Lifespan Management (Load Pickle Model on Startup)
# ---------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
model_pipeline: Any = None
model_metadata: Dict[str, Any] = {}

DEFAULT_TARGET_NAMES = ["setosa", "versicolor", "virginica"]
DEFAULT_FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_model_artifact(file_path: Path):
    global model_pipeline, model_metadata
    if not file_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {file_path}")

    with open(file_path, "rb") as f:
        artifact = pickle.load(f)

    if isinstance(artifact, dict) and "model" in artifact:
        model_pipeline = artifact["model"]
        model_metadata = {
            "feature_names": artifact.get("feature_names", DEFAULT_FEATURE_NAMES),
            "target_names": artifact.get("target_names", DEFAULT_TARGET_NAMES),
            "description": artifact.get("description", "Trained Scikit-Learn Model Pipeline")
        }
    else:
        # Fallback if raw estimator was saved
        model_pipeline = artifact
        model_metadata = {
            "feature_names": DEFAULT_FEATURE_NAMES,
            "target_names": DEFAULT_TARGET_NAMES,
            "description": "Trained Scikit-Learn Estimator"
        }
    print(f"[INFO] Successfully loaded model from {file_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the serialized pickle model
    try:
        load_model_artifact(MODEL_PATH)
    except Exception as e:
        print(f"[WARNING] Model could not be loaded on startup: {e}")
        model_pipeline = None
    yield
    # Shutdown: Clean up resources if necessary
    print("[INFO] Application shutting down.")


# ---------------------------------------------------------
# FastAPI App Initialization & OpenAPI Documentation
# ---------------------------------------------------------

app = FastAPI(
    title="Iris Flower Species Prediction API",
    description="""
## SML & BDA Lab - Experiment 7: Model Packaging & Deployment
This REST API serves predictions from a Scikit-Learn **Random Forest Classifier** trained on the famous Iris Dataset.
The model was packaged and serialized using `pickle` (`model.pkl`) and is deployed via **FastAPI** & **Uvicorn**.

### Key Features:
* **/predict**: Single-sample flower species inference with confidence scores and class probability breakdown.
* **/predict/batch**: High-throughput batch inference endpoint for multiple flower samples.
* **/model-info**: Inspection endpoint providing architecture and metadata details of the loaded model.
* **/health**: Liveness and readiness probe for orchestration/monitoring.
* **/docs**: Interactive Swagger UI with try-it-out capabilities.
* **/redoc**: Alternative API documentation view.
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Manan Chahal",
        "email": "mananchahal01@gmail.com"
    },
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external frontend or dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get(
    "/",
    tags=["Root"],
    summary="API Root Information",
    description="Returns a welcome message, API status, and links to documentation."
)
def root():
    return {
        "message": "Welcome to the Iris Flower Species Prediction API (Experiment 7)",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "model_info": "/model-info",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch"
        }
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health & Readiness Probe",
    description="Checks whether the API is running and the serialized model is loaded and ready to serve requests."
)
def health_check():
    is_loaded = model_pipeline is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_artifact": str(MODEL_PATH.name),
        "version": app.version
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model Info"],
    summary="Get Model Architecture & Metadata",
    description="Retrieves the feature requirements, target classes, and description of the deployed model artifact."
)
def get_model_info():
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )

    model_name = type(model_pipeline).__name__
    if hasattr(model_pipeline, "named_steps") and "classifier" in model_pipeline.named_steps:
        model_name = type(model_pipeline.named_steps["classifier"]).__name__

    return ModelInfoResponse(
        model_name=model_name,
        feature_names=model_metadata.get("feature_names", DEFAULT_FEATURE_NAMES),
        target_names=model_metadata.get("target_names", DEFAULT_TARGET_NAMES),
        description=model_metadata.get("description", "Iris Classifier"),
        status="Loaded and operational"
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict Single Iris Flower Species",
    description="Takes four morphological measurements (sepal length/width, petal length/width) and returns predicted species with class probabilities."
)
def predict(
    features: IrisFeatures = Body(
        ...,
        openapi_examples=PREDICT_EXAMPLES
    )
):
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded on the server."
        )

    # Format incoming data into a DataFrame with required feature column names
    feature_names = model_metadata.get("feature_names", DEFAULT_FEATURE_NAMES)
    input_data = pd.DataFrame([{
        "sepal_length": features.sepal_length,
        "sepal_width": features.sepal_width,
        "petal_length": features.petal_length,
        "petal_width": features.petal_width
    }])[feature_names]

    try:
        class_id = int(model_pipeline.predict(input_data)[0])
        probabilities = model_pipeline.predict_proba(input_data)[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing model inference: {str(e)}"
        )

    target_names = model_metadata.get("target_names", DEFAULT_TARGET_NAMES)
    class_name = target_names[class_id] if class_id < len(target_names) else f"class_{class_id}"

    prob_dict = {
        name: round(float(prob), 4)
        for name, prob in zip(target_names, probabilities)
    }

    return PredictionResponse(
        predicted_class_id=class_id,
        predicted_class_name=class_name,
        confidence=round(float(probabilities[class_id]), 4),
        probabilities=prob_dict,
        input_features=features
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch Prediction for Multiple Iris Samples",
    description="Takes an array of sample measurements and executes batch inference."
)
def predict_batch(
    batch: BatchIrisFeatures = Body(
        ...,
        openapi_examples=BATCH_PREDICT_EXAMPLES
    )
):
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded on the server."
        )

    feature_names = model_metadata.get("feature_names", DEFAULT_FEATURE_NAMES)
    target_names = model_metadata.get("target_names", DEFAULT_TARGET_NAMES)

    rows = [
        {
            "sepal_length": item.sepal_length,
            "sepal_width": item.sepal_width,
            "petal_length": item.petal_length,
            "petal_width": item.petal_width
        }
        for item in batch.samples
    ]
    input_df = pd.DataFrame(rows)[feature_names]

    try:
        class_ids = model_pipeline.predict(input_df)
        probabilities = model_pipeline.predict_proba(input_df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing batch inference: {str(e)}"
        )

    results: List[PredictionResponse] = []
    for i, (cid, probs) in enumerate(zip(class_ids, probabilities)):
        cid_int = int(cid)
        cname = target_names[cid_int] if cid_int < len(target_names) else f"class_{cid_int}"
        p_dict = {
            name: round(float(prob), 4)
            for name, prob in zip(target_names, probs)
        }
        results.append(
            PredictionResponse(
                predicted_class_id=cid_int,
                predicted_class_name=cname,
                confidence=round(float(probs[cid_int]), 4),
                probabilities=p_dict,
                input_features=batch.samples[i]
            )
        )

    return BatchPredictionResponse(
        total_samples=len(results),
        predictions=results
    )


# ---------------------------------------------------------
# Execution Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
