"""Train and save a CIFAR-10 CNN."""

import os
import json
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.labels import CIFAR10_CLASSES


MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "cifar10_model.keras"))
ROWS_API = "https://datasets-server.huggingface.co/rows"
PARQUET_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/refs%2Fconvert%2Fparquet/plain_text"
DATA_DIR = ROOT / "data" / "cifar10"


def read_url(url):
    """Fetch a dataset API response or image with a small transient retry."""
    request = Request(url, headers={"User-Agent": "CIFAR-Vision-AI/1.0"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(5 * (attempt + 1), 30)
                time.sleep(delay)
            elif attempt == 4:
                raise
            else:
                time.sleep(0.5 * (attempt + 1))
        except (URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def download_balanced_split(split, samples_per_class):
    """Download a balanced CIFAR-10 subset from individual hosted rows."""
    counts = [0] * len(CIFAR10_CLASSES)
    image_rows = []
    offset = 0
    while min(counts) < samples_per_class:
        if offset:
            time.sleep(2)
        params = urlencode({
            "dataset": "uoft-cs/cifar10",
            "config": "plain_text",
            "split": split,
            "offset": offset,
            "length": 100,
        })
        page = json.loads(read_url(f"{ROWS_API}?{params}"))
        rows = page.get("rows", [])
        if not rows:
            raise RuntimeError(f"Could not collect a balanced CIFAR-10 {split} subset.")
        for item in rows:
            row = item["row"]
            label = int(row["label"])
            if counts[label] < samples_per_class:
                image_rows.append((label, row["img"]["src"]))
                counts[label] += 1
        offset += len(rows)
        print(f"{split}: collected {min(counts)}/{samples_per_class} per class", flush=True)

    def decode_image(item):
        label, image_url = item
        with Image.open(BytesIO(read_url(image_url))) as image:
            image = image.convert("RGB")
            if image.size != (32, 32):
                raise ValueError(f"Expected a 32x32 CIFAR-10 image, received {image.size}.")
            return np.asarray(image, dtype=np.uint8), label

    with ThreadPoolExecutor(max_workers=24) as executor:
        samples = list(executor.map(decode_image, image_rows))
    images = np.stack([sample[0] for sample in samples]).astype("float32") / 255.0
    labels = np.asarray([sample[1] for sample in samples], dtype="int64")
    return images, labels


def download_parquet_split(split):
    """Download one complete official CIFAR-10 parquet split into the local cache."""
    destination = DATA_DIR / f"{split}-0000.parquet"
    if destination.is_file():
        return destination

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".parquet.part")
    url = f"{PARQUET_BASE}/{split}/0000.parquet"
    request = Request(url, headers={"User-Agent": "CIFAR-Vision-AI/1.0"})
    with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
        total_bytes = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                print(f"Downloading {split} split: {downloaded / total_bytes:.0%}", flush=True)
    temporary_path.replace(destination)
    return destination


def load_parquet_split(split):
    """Decode every row of a full CIFAR-10 parquet split into model arrays."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(download_parquet_split(split), columns=["img", "label"])
    image_values = table.column("img").combine_chunks().field("bytes").to_pylist()
    labels = table.column("label").to_numpy().astype("int64", copy=False)
    images = np.empty((len(image_values), 32, 32, 3), dtype="uint8")

    def decode_image(image_bytes):
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            if image.size != (32, 32):
                raise ValueError(f"Expected a 32x32 CIFAR-10 image, received {image.size}.")
            return np.asarray(image, dtype="uint8")

    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, pixels in enumerate(executor.map(decode_image, image_values)):
            images[index] = pixels

    del image_values, table
    gc.collect()
    return images.astype("float32") / 255.0, labels


def build_model():
    """Build a compact convolutional classifier for 32x32 RGB images."""
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(), layers.Dropout(0.2),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(), layers.Dropout(0.3),
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.35),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    data_source = os.getenv("CIFAR10_SOURCE", "huggingface").lower()
    if data_source == "huggingface_subset":
        train_images, train_labels = download_balanced_split(
            "train", int(os.getenv("TRAIN_SAMPLES_PER_CLASS", "500"))
        )
        test_images, test_labels = download_balanced_split(
            "test", int(os.getenv("TEST_SAMPLES_PER_CLASS", "100"))
        )
        validation_data = (test_images, test_labels)
    elif data_source == "keras":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        train_images = train_images.astype("float32") / 255.0
        test_images = test_images.astype("float32") / 255.0
        validation_data = (test_images, test_labels)
    else:
        train_images, train_labels = load_parquet_split("train")
        test_images, test_labels = load_parquet_split("test")
        validation_data = (test_images, test_labels)

    model = build_model()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ]
    model.fit(
        train_images, train_labels,
        epochs=int(os.getenv("EPOCHS", "40")), batch_size=64,
        callbacks=callbacks, validation_data=validation_data,
    )
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()