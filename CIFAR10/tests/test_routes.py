from io import BytesIO

import numpy as np
from PIL import Image

from app import create_app


class FakeModelService:
    is_loaded = True

    def predict(self, _batch):
        return np.eye(1, 10, 5, dtype=np.float32)[0]


def make_image():
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_home_page_and_health_endpoint(tmp_path):
    client = create_app({"TESTING": True, "MODEL_SERVICE": FakeModelService(), "LOG_PATH": tmp_path / "test.log"}).test_client()

    assert client.get("/").status_code == 200
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json == {"status": "healthy", "model_loaded": True}


def test_predict_endpoint_accepts_image_and_returns_top_three(tmp_path):
    client = create_app({"TESTING": True, "MODEL_SERVICE": FakeModelService(), "LOG_PATH": tmp_path / "test.log"}).test_client()

    response = client.post("/api/predict", data={"image": (make_image(), "sample.png")})

    assert response.status_code == 200
    assert response.json["prediction"] == {"class": "dog", "confidence": 1.0}
    assert len(response.json["top_predictions"]) == 3


def test_predict_endpoint_rejects_invalid_image(tmp_path):
    client = create_app({"TESTING": True, "MODEL_SERVICE": FakeModelService(), "LOG_PATH": tmp_path / "test.log"}).test_client()

    response = client.post("/api/predict", data={"image": (BytesIO(b"not an image"), "fake.jpg")})

    assert response.status_code == 400
    assert response.json["success"] is False


def test_health_is_unhealthy_without_a_model(tmp_path):
    client = create_app({"TESTING": True, "MODEL_SERVICE": type("MissingModel", (), {"is_loaded": False})(), "LOG_PATH": tmp_path / "test.log"}).test_client()

    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json["model_loaded"] is False


def test_app_starts_and_reports_missing_model(tmp_path):
    client = create_app({
        "TESTING": True,
        "MODEL_PATH": tmp_path / "missing.keras",
        "LOG_PATH": tmp_path / "test.log",
    }).test_client()

    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json == {"status": "unhealthy", "model_loaded": False}