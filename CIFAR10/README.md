# CIFAR Vision AI

A local Flask application for classifying uploaded images against the ten CIFAR-10 categories. It preprocesses images in memory, displays a prediction and its top three class probabilities, and includes scripts to train and evaluate a Keras CNN.

## Features

- Drag-and-drop or browse for JPG, PNG, and WEBP images up to 5 MB.
- Responsive preview, dimensions, file size, clear/reset, and prediction states.
- In-memory image processing and a 32 × 32 RGB preprocessing pipeline.
- Sorted top-three probabilities and a visual confidence indicator.
- Health endpoint, structured logging, friendly errors, and pytest coverage.
- Includes a trained CIFAR-10 checkpoint at `models/cifar10_model.keras` (75.44% accuracy on the 10,000-image test split).
- TensorFlow model is loaded once at app startup. If the checkpoint is removed, health reports unhealthy and predictions return 503 until weights are available.

## Architecture

Flask uses an application factory in `app/__init__.py`; page and API endpoints are separate blueprints. `PredictionService` coordinates validation, preprocessing, inference, and response formatting. `ModelService` owns the in-memory model. CIFAR-10 labels live in one module and are shared with the page template.

```text
cifar10-flask-app/
├── app/
│   ├── ml/                 # Labels and image preprocessing
│   ├── routes/             # Page and JSON API blueprints
│   ├── services/           # Model and prediction services
│   ├── static/             # CSS and browser JavaScript
│   ├── templates/          # Jinja pages
│   └── utils/              # Validation and logging
├── config/                 # Environment-backed settings
├── logs/                   # Rotating application logs
├── models/                 # Place cifar10_model.keras here
├── scripts/                # Model training and evaluation
├── tests/                  # Flask, prediction, and preprocessing tests
├── uploads/                 # Reserved; uploads are not persisted
├── requirements.txt
└── run.py
```

## Stack and dataset

Python 3.11 or 3.12, Flask, TensorFlow/Keras, NumPy, Pillow, python-dotenv, and pytest. CIFAR-10 contains 60,000 32 × 32 colour images in airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck classes. Arbitrary high-resolution photos may not resemble the training data; images are resized to the model's 32 × 32 input.

## Setup on Windows PowerShell

From this project directory:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

If script activation is blocked, use the environment's Python directly, for example `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\python.exe run.py`, or allow scripts for this PowerShell process with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

TensorFlow's native Windows wheel availability depends on the Python/TensorFlow version. If pip cannot install TensorFlow on your Windows setup, use a supported TensorFlow Python environment or WSL2; Flask can run without TensorFlow as long as no model file is present.

## Train and evaluate

Training uses all 50,000 CIFAR-10 training images and the 10,000-image test split for validation, normalizes the data, trains the CNN with validation callbacks, and saves the best model:

```powershell
python scripts/train_model.py
```

The default downloads and caches the complete train/test parquet splits under `data/cifar10/`. To use the TensorFlow/Keras archive source instead, set `$env:CIFAR10_SOURCE = "keras"` before running the script. For a balanced hosted subset instead of all 50,000 images, set:

```powershell
$env:CIFAR10_SOURCE = "huggingface_subset"
$env:TRAIN_SAMPLES_PER_CLASS = "500"
$env:TEST_SAMPLES_PER_CLASS = "100"
$env:EPOCHS = "10"
python scripts/train_model.py
```

This mode fetches individual 32 × 32 examples from the CIFAR-10 test/train splits and is useful for a quicker local model build. The default remains the full Keras dataset.

Set training duration when needed, for example `$env:EPOCHS = "5"` for a short smoke run. Evaluate the resulting model:

```powershell
python scripts/evaluate_model.py
```

To export one authentic CIFAR-10 test image per class for trying the upload flow:

```powershell
python scripts/export_sample_images.py
```

The ten PNGs are saved under `app/static/samples/` and appear as downloads in the interface.

The repository includes the trained model at `models/cifar10_model.keras`. It accepts normalized `(32, 32, 3)` RGB batches and returns ten probabilities in canonical CIFAR-10 class order. `MODEL_PATH` can point to another compatible model file.

## Run the app

```powershell
python run.py
```

Open http://127.0.0.1:5000. Uploaded image bytes are processed in memory and are not saved.

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Settings include `SECRET_KEY`, `MODEL_PATH`, `MAX_UPLOAD_SIZE` (default 5,242,880 bytes), `LOG_PATH`, and `EPOCHS`. The application factory loads this file with python-dotenv; variables can also be set directly in PowerShell. Do not commit `.env` or cached training data.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Classification interface |
| `GET` | `/api/health` | Model readiness; returns 200 when loaded, otherwise 503 |
| `POST` | `/api/predict` | Image prediction via multipart form field `image` |

Successful predictions return `success`, the predicted class and probability from 0 to 1, and up to three descending class/probability objects. Errors return `success: false`, a readable error, and an appropriate HTTP status. Oversized request bodies return 413.

Example PowerShell request:

```powershell
curl.exe -F "image=@C:\path\to\image.jpg" http://127.0.0.1:5000/api/predict
```

## Tests

```powershell
python -m pytest
```

Tests cover RGB conversion, 32 × 32 resizing, normalization, batching, top-three ordering, health state, page rendering, and valid/invalid prediction requests. Test doubles avoid requiring TensorFlow weights.

## Troubleshooting

- **Health is unhealthy:** train a model or place a compatible model at `models/cifar10_model.keras`; check `logs/app.log` for load errors.
- **TensorFlow import/install error:** verify the selected Python version and platform support, or use WSL2.
- **Unsupported file or corrupt image:** provide a real JPG/JPEG, PNG, or WEBP image under 5 MB.
- **Port 5000 is in use:** stop the other process or change the port in `run.py`.

## Possible extensions

Model benchmarking, a confusion matrix report, batched inference, and optional confidence calibration can be added without changing the upload contract.