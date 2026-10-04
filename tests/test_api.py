import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import app  # noqa: E402
from train import MODEL_DIR, train  # noqa: E402

SAMPLE = json.loads((ROOT / "data" / "sample_request.json").read_text())


@pytest.fixture(scope="session")
def client():
    if not (MODEL_DIR / "cancer_model.pkl").exists():
        train()
    return TestClient(app)


def test_health(client):
    assert client.get("/").json() == {"status": "healthy"}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["n_features"] == 30 and "roc_auc" in body["metrics"]


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert r.json()["diagnosis"] in {"malignant", "benign"}


def test_predict_rejects_negative(client):
    assert client.post("/predict", json={**SAMPLE, "mean_radius": -1}).status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "mean_radius"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"samples": [SAMPLE, SAMPLE]})
    assert r.status_code == 200 and r.json()["count"] == 2
