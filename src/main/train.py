# -*- coding: utf-8 -*-
import logging
import os

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score

from src.utils.load_config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_preprocessed_data(file_path):
    """Load preprocessed data from a file."""
    logger.info(f"Loading preprocessed data from {file_path}")
    data = np.load(file_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, kernel=None):
    """Train the Gaussian Process Classifier model."""
    if kernel is None:
        kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel)
    clf.fit(X_train, y_train.ravel())
    return clf


def evaluate_model(clf, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {score}")
    return score


def save_model(clf, output_path):
    """Save the trained model to a file."""

    os.makedirs("models", exist_ok=True)
    logger.info(f"Directory models is ready.")
    joblib.dump(clf, output_path)
    logger.info(f"Model saved to {output_path}")


def main(config_path):
    """Main function to train the model."""
    # Load configurations
    config = load_config(config_path)

    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data(
        config["data"]["preprocessed_data_path"]
    )

    # Train the model
    clf = train_model(X_train, y_train)

    # Evaluate the model
    score = evaluate_model(clf, X_test, y_test)

    # Save the trained model
    save_model(clf, config["model"]["model_output_path"])


if __name__ == "__main__":
    main("config.yml")
