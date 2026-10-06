"""Upload validation helpers."""

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def validate_image(filename, image_bytes, max_bytes=MAX_IMAGE_BYTES):
    """Validate file extension, size, and actual image contents."""
    if not image_bytes:
        raise ValueError("Please select an image.")
    if len(image_bytes) > max_bytes:
        raise ValueError("File size exceeds the 5 MB limit.")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file format. Please upload JPG, PNG, or WEBP.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Unsupported file format. Please upload JPG, PNG, or WEBP.")
            image.verify()
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Unable to process this image.") from error