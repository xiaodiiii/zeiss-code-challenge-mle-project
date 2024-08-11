# -*- coding: utf-8 -*-
import logging

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.load_config import load_config
from src.utils.load_data import load_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load the trained model from a file."""
    logger.info(f"Loading model from {model_path}")
    clf = joblib.load(model_path)
    return clf


def preprocess_inference_data(X_inference):
    """Standardize the inference data."""
    logger.info("Standardizing inference data")
    X_inference = StandardScaler().fit_transform(X_inference)
    return X_inference


def predict(clf, X_inference):
    """Predict labels using the trained model."""
    logger.info("Predicting labels")
    y_pred = clf.predict_proba(X_inference)[:, 1]
    return y_pred


def save_predictions(inference_df, y_pred, output_path):
    """Save the predictions to a CSV file."""
    logger.info(f"Saving predictions to {output_path}")
    inference_df = inference_df.assign(y_pred=np.round(y_pred, 0))
    inference_df.to_csv(output_path, index=False)


def main(config_path):
    """Main function to load the model, make predictions, and save them."""
    # Load configurations
    config = load_config(config_path)

    # Load the trained model
    clf = load_model(config["model"]["model_output_path"])

    # Load and preprocess inference data
    inference_df = load_data(config["data"]["inference_data_path"])
    X_inference = inference_df.values
    X_inference = preprocess_inference_data(X_inference)

    # Predict labels
    y_pred = predict(clf, X_inference)

    # Save the predictions
    save_predictions(inference_df, y_pred, config["data"]["prediction_output_path"])


if __name__ == "__main__":
    main("config.yml")
