"""Evaluate the saved model against the CIFAR-10 test split."""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "cifar10_model.keras"))


def main():
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Run python scripts/train_model.py first.")
    (_, _), (images, labels) = tf.keras.datasets.cifar10.load_data()
    model = load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(images.astype("float32") / 255.0, labels, verbose=0)
    print("CIFAR-10 Model Evaluation")
    print("-------------------------")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()