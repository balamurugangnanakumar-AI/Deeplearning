"""Export one labeled PNG example per CIFAR-10 class for app users."""

import json
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.labels import CIFAR10_CLASSES


SAMPLE_DIR = ROOT / "app" / "static" / "samples"
ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET_PARAMS = {
    "dataset": "uoft-cs/cifar10",
    "config": "plain_text",
    "split": "test",
    "length": 100,
}


def read_url(url):
    request = Request(url, headers={"User-Agent": "CIFAR-Vision-AI/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def main():
    """Find and save one real test-set image for each class."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    found = set()
    offset = 0

    while len(found) < len(CIFAR10_CLASSES):
        params = {**DATASET_PARAMS, "offset": offset}
        payload = json.loads(read_url(f"{ROWS_API}?{urlencode(params)}"))
        rows = payload.get("rows", [])
        if not rows:
            raise RuntimeError("The CIFAR-10 test split ended before all classes were found.")

        for item in rows:
            row = item["row"]
            class_name = CIFAR10_CLASSES[row["label"]]
            if class_name in found:
                continue

            with Image.open(BytesIO(read_url(row["img"]["src"]))) as image:
                image = image.convert("RGB")
                if image.size != (32, 32):
                    raise ValueError(f"Expected a 32x32 CIFAR-10 image, received {image.size}.")
                output_path = SAMPLE_DIR / f"{class_name}.png"
                image.save(output_path, format="PNG", optimize=True)

            found.add(class_name)
            print(f"Saved {output_path.relative_to(ROOT)}")
            if len(found) == len(CIFAR10_CLASSES):
                break

        offset += len(rows)


if __name__ == "__main__":
    main()