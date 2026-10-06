"""Load and retain a Keras model for the lifetime of the application."""

from pathlib import Path
from threading import Lock

import numpy as np


class ModelService:
    """Small wrapper around a lazily imported TensorFlow model."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._model = None
        self._lock = Lock()

    @property
    def is_loaded(self):
        return self._model is not None

    def load_model(self):
        """Load model weights once; leave the app usable if no model is present."""
        if self.is_loaded:
            return True
        if not self.model_path.is_file():
            import logging

            logging.getLogger("cifar_vision").warning(
                "Model file not found at %s; prediction is disabled", self.model_path.name
            )
            return False

        try:
            from tensorflow.keras.models import load_model

            self._model = load_model(self.model_path)
        except Exception:
            import logging

            logging.getLogger("cifar_vision").exception("Could not load the CIFAR-10 model")
            self._model = None
            return False

        import logging

        logging.getLogger("cifar_vision").info("Model loaded successfully")
        return True

    def predict(self, batch):
        """Return the model's class probabilities for one image batch."""
        if not self.is_loaded:
            raise RuntimeError("The CIFAR-10 model is not loaded.")
        with self._lock:
            output = self._model.predict(batch, verbose=0)
        probabilities = np.asarray(output, dtype=np.float32)
        if probabilities.shape != (1, 10) or not np.isfinite(probabilities).all():
            raise RuntimeError("The model returned invalid prediction scores.")
        return probabilities[0]