"""Page routes."""

from flask import Blueprint, render_template

from app.ml.labels import CIFAR10_CLASSES


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    return render_template(
        "index.html", classes=CIFAR10_CLASSES, sample_images=CIFAR10_CLASSES
    )