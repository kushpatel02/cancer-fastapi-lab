# FastAPI Lab — Breast Cancer Diagnosis API 🩺

Modified version of the [FastAPI Lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) from Dr. Ramin Mohammadi's MLOps course.

## Changes from the original lab

| Area | Original | This version |
|---|---|---|
| Dataset | Iris (4 features, 3 classes) | Wisconsin Breast Cancer (30 features, binary) |
| Model | Decision Tree | `StandardScaler` + `GradientBoostingClassifier` pipeline |
| Evaluation | None saved | Accuracy, F1, ROC-AUC saved to `model/metadata.json` |
| Request schema | Hand-written 4-field Pydantic model | Generated from the feature list with `create_model`; all values validated `>= 0` |
| Response | Class id | Diagnosis, confidence, per-class probabilities (typed `response_model`) |
| Endpoints | `/`, `/predict` | `/`, `/model/info`, `/predict`, `/predict/batch` (up to 500 samples) |
| Errors | — | 503 if model not trained, 422 on invalid/missing fields |
| Testing | — | `pytest` suite using `TestClient` |
| Deployment | Local only | `Dockerfile` |

## Structure
```
cancer-fastapi-lab/
├── data/sample_request.json
├── model/                 # created by train.py
├── src/  data.py  train.py  predict.py  main.py
├── tests/test_api.py
├── Dockerfile
└── requirements.txt
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd src
python train.py
uvicorn main:app --reload
```
Open http://localhost:8000/docs → `/predict` → "Try it out" → paste `data/sample_request.json`.

## Test
```bash
pytest -v        # from repo root
```

## Docker
```bash
docker build -t cancer-api .
docker run -p 8000:8000 cancer-api
```
