# Experiment 7: Machine Learning Model Packaging, Serialization & Serving with FastAPI and Uvicorn

**Course:** Scalable Machine Learning & Big Data Analytics (SML & BDA) / MLOps  
**Student Name:** Manan Chahal  
**Roll Number:** 23102C0044  

---

## 📌 Overview
This project demonstrates the complete end-to-end MLOps pipeline for:
1. **Model Training & Evaluation:** Training a Scikit-Learn `Pipeline` (`StandardScaler` + `RandomForestClassifier`) on the Iris flower dataset.
2. **Model Serialization:** Packaging the trained model and associated feature/target metadata into a pickle artifact (`model.pkl`).
3. **Model Serving & Deployment:** Building a high-performance REST API using **FastAPI** (`main.py`) and serving it with the **Uvicorn** ASGI server.
4. **Interactive Documentation:** Automatic OpenAPI documentation available at `/docs` (Swagger UI) and `/redoc` (ReDoc) with schema validation and example payloads.

---

## 📁 Repository Structure
```
Experiment 7/
├── .venv/                  # Python virtual environment (uv managed)
├── pyproject.toml          # Project dependencies & configurations
├── uv.lock                 # Pinned dependencies lockfile
├── train_model.ipynb       # Well-formatted Jupyter notebook with training & uvicorn cell
├── model.pkl               # Serialized ML pipeline + metadata artifact
├── main.py                 # FastAPI application with /predict, batch inference, docs
└── README.md               # Documentation and execution instructions
```

---

## 🚀 Getting Started

### 1. Environment Setup
The project uses `uv` for fast, reproducible environment management:
```powershell
# Sync dependencies
uv sync
```

Alternatively using the virtual environment directly:
```powershell
.\.venv\Scripts\activate
```

---

### 2. Training the Model (`train_model.ipynb`)
Open `train_model.ipynb` in VS Code or Jupyter Notebook:
- Execute cells sequentially to inspect the dataset, train the `StandardScaler` + `RandomForestClassifier` pipeline, evaluate metrics (Accuracy: **96.67%**), and serialize the model into `model.pkl`.
- The final cell contains:
  ```python
  import uvicorn

  if __name__ == "__main__":
      uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
  ```
  Running this last cell will start the FastAPI server directly inside the notebook.

---

### 3. Running the FastAPI Server
To launch the API server from your terminal:

```powershell
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Or directly using Python:
```powershell
python main.py
```

Server endpoints:
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* **Model Info:** [http://127.0.0.1:8000/model-info](http://127.0.0.1:8000/model-info)

---

## 📡 API Endpoints & Examples

### 1. Single Prediction (`POST /predict`)
Predicts the species for a single flower sample with confidence score and class probability distribution.

#### Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

#### Response:
```json
{
  "predicted_class_id": 0,
  "predicted_class_name": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "input_features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

---

### 2. Batch Prediction (`POST /predict/batch`)
Allows batch scoring for multiple records in a single network round-trip.

#### Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "samples": [
         {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
         {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
         {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
       ]
     }'
```

#### Response:
```json
{
  "total_samples": 3,
  "predictions": [
    {
      "predicted_class_id": 0,
      "predicted_class_name": "setosa",
      "confidence": 1.0,
      "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
      },
      "input_features": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      }
    },
    {
      "predicted_class_id": 1,
      "predicted_class_name": "versicolor",
      "confidence": 0.9778,
      "probabilities": {
        "setosa": 0.0007,
        "versicolor": 0.9778,
        "virginica": 0.0214
      },
      "input_features": {
        "sepal_length": 6.0,
        "sepal_width": 2.9,
        "petal_length": 4.5,
        "petal_width": 1.5
      }
    },
    {
      "predicted_class_id": 2,
      "predicted_class_name": "virginica",
      "confidence": 0.9981,
      "probabilities": {
        "setosa": 0.0001,
        "versicolor": 0.0018,
        "virginica": 0.9981
      },
      "input_features": {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3
      }
    }
  ]
}
```

---

### 3. Health & Readiness Probe (`GET /health`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_artifact": "model.pkl",
  "version": "1.0.0"
}
```

---

## 🧪 Testing with Python
You can test the API using Python's `httpx` or `requests`:
```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "sepal_length": 5.8,
    "sepal_width": 2.7,
    "petal_length": 5.1,
    "petal_width": 1.9
}

response = requests.post(url, json=payload)
print(response.json())
```
