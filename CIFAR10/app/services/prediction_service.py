"""Image-to-class prediction orchestration."""

import logging

import numpy as np

from app.ml.labels import CIFAR10_CLASSES
from app.ml.preprocessing import preprocess_image
from app.utils.validators import validate_image


class PredictionService:
    """Validate an upload, run the model, and format its strongest classes."""

    def __init__(self, model_service, max_upload_size=5 * 1024 * 1024):
        self.model_service = model_service
        self.max_upload_size = max_upload_size
        self.logger = logging.getLogger("cifar_vision")

    def predict(self, filename, image_bytes):
        self.logger.info("Prediction request received")
        validate_image(filename, image_bytes, self.max_upload_size)
        batch = preprocess_image(image_bytes)
        probabilities = np.asarray(self.model_service.predict(batch), dtype=np.float64)
        if probabilities.shape != (10,) or not np.isfinite(probabilities).all():
            raise RuntimeError("The model returned invalid prediction scores.")

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {"class": CIFAR10_CLASSES[index], "confidence": round(float(probabilities[index]), 4)}
            for index in top_indices
        ]
        prediction = top_predictions[0]
        self.logger.info(
            "Prediction: %s | confidence=%.4f", prediction["class"], prediction["confidence"]
        )
        return {"prediction": prediction, "top_predictions": top_predictions}