"""Shared preprocessing for uploaded CIFAR-10 prediction images."""

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


def preprocess_image(image_bytes):
    """Decode image bytes to one normalized 32x32 RGB batch."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Unable to process this image.") from error
    return np.expand_dims(pixels, axis=0)