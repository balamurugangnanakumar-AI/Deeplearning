from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.services.prediction_service import PredictionService
from app.utils.validators import validate_image


class FakeModelService:
    def predict(self, _batch):
        return np.array([0.01, 0.02, 0.03, 0.08, 0.05, 0.72, 0.04, 0.03, 0.01, 0.01])


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (24, 18), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_prediction_returns_sorted_top_three():
    result = PredictionService(FakeModelService()).predict("test.jpg", image_bytes())

    assert result["prediction"] == {"class": "dog", "confidence": 0.72}
    assert [item["class"] for item in result["top_predictions"]] == ["dog", "cat", "deer"]


@pytest.mark.parametrize("image_format,filename", [("JPEG", "image.jpg"), ("PNG", "image.png"), ("WEBP", "image.webp")])
def test_validator_accepts_supported_images(image_format, filename):
    buffer = BytesIO()
    Image.new("RGB", (12, 12), "blue").save(buffer, format=image_format)

    validate_image(filename, buffer.getvalue())


def test_validator_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_image("image.gif", image_bytes())


def test_validator_rejects_oversized_image():
    with pytest.raises(ValueError, match="5 MB"):
        validate_image("image.jpg", b"x" * 11, max_bytes=10)


def test_validator_rejects_corrupt_image():
    with pytest.raises(ValueError, match="Unable to process"):
        validate_image("image.jpg", b"not an image")