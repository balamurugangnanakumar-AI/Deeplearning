"""Flask application factory for CIFAR Vision AI."""

import logging

from flask import Flask, jsonify, render_template, request

from app.routes.main import main_bp
from app.routes.prediction import prediction_bp
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.utils.logger import configure_logging
from config.settings import Settings


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Settings)
    if test_config:
        app.config.update(test_config)

    configure_logging(app.config["LOG_PATH"])
    model_service = app.config.get("MODEL_SERVICE") or ModelService(app.config["MODEL_PATH"])
    if "MODEL_SERVICE" not in app.config:
        model_service.load_model()
    app.extensions["model_service"] = model_service
    app.extensions["prediction_service"] = PredictionService(
        model_service, app.config["MAX_CONTENT_LENGTH"]
    )
    logging.getLogger("cifar_vision").info("CIFAR Vision AI application startup complete")

    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)

    @app.errorhandler(413)
    def request_too_large(_error):
        message = "File size exceeds the 5 MB limit."
        if request.path.startswith("/api/"):
            return jsonify(success=False, error=message), 413
        return render_template("error.html", message=message), 413

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify(success=False, error="Endpoint not found."), 404
        return render_template("error.html", message="That page could not be found."), 404

    @app.errorhandler(400)
    def bad_request(_error):
        if request.path.startswith("/api/"):
            return jsonify(success=False, error="The request could not be understood."), 400
        return render_template("error.html", message="The request could not be understood."), 400

    @app.errorhandler(500)
    def internal_error(_error):
        app.logger.exception("Unhandled application error")
        if request.path.startswith("/api/"):
            return jsonify(success=False, error="Unable to process this request."), 500
        return render_template("error.html", message="Something went wrong. Please try again."), 500

    return app