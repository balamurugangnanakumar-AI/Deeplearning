from io import BytesIO

import numpy as np
from PIL import Image

from app.ml.preprocessing import preprocess_image


def test_preprocessing_converts_resizes_normalizes_and_batches_rgb():
    source = Image.new("RGBA", (80, 40), (255, 128, 0, 100))
    buffer = BytesIO()
    source.save(buffer, format="PNG")

    result = preprocess_image(buffer.getvalue())

    assert result.shape == (1, 32, 32, 3)
    assert result.dtype == np.float32
    assert np.allclose(result[0, 16, 16], [1.0, 128 / 255, 0.0], atol=0.02)
    assert result.min() >= 0.0
    assert result.max() <= 1.0