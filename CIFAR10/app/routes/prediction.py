"""Prediction and model health API routes."""

from flask import Blueprint, current_app, jsonify, request


prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


@prediction_bp.get("/health")
def health():
    model_loaded = current_app.extensions["model_service"].is_loaded
    status = "healthy" if model_loaded else "unhealthy"
    return jsonify(status=status, model_loaded=model_loaded), 200 if model_loaded else 503


@prediction_bp.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(success=False, error="Please select an image."), 400

    uploaded_file = request.files["image"]
    if not uploaded_file.filename:
        return jsonify(success=False, error="Please select an image."), 400

    try:
        result = current_app.extensions["prediction_service"].predict(
            uploaded_file.filename, uploaded_file.read()
        )
    except ValueError as error:
        current_app.logger.warning("Image upload rejected: %s", error)
        return jsonify(success=False, error=str(error)), 400
    except RuntimeError:
        current_app.logger.exception("Prediction could not be completed")
        return jsonify(success=False, error="The model is unavailable. Train or install a model first."), 503
    except Exception:
        current_app.logger.exception("Unexpected prediction failure")
        return jsonify(success=False, error="Unable to process this image."), 500

    return jsonify(success=True, **result)