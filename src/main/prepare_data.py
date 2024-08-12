# -*- coding: utf-8 -*-
"""Data preparation script."""
# -*- coding: utf-8 -*-
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.load_config import load_config
from src.utils.load_data import load_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(df, features, target):
    """Preprocess the data: split into
    features and labels, then scale the features.
    """
    x = df[features].values
    y = df[target].values
    logger.info(f"Data shape: X={x.shape}, y={y.shape}")

    # Standardize features
    x = StandardScaler().fit_transform(x)
    return x, y


def split_data(x, y, test_size, random_state):
    """Split the data into training and test sets."""
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def save_preprocessed_data(x_train, x_test, y_train, y_test, output_path):
    """Save preprocessed data to a file."""
    np.savez(
        output_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )
    logger.info(f"Preprocessed data saved to {output_path}")


def main(config_path):
    """Main function to load and preprocess data."""
    # Load configurations
    config = load_config(config_path)

    # Load and preprocess data
    df = load_data(file_path=config["data"]["raw_data_path"])
    x, y = preprocess_data(df, config["features"], config["target"])

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = split_data(
        x,
        y,
        config["preprocessing"]["test_size"],
        config["preprocessing"]["random_state"],
    )

    # Save the preprocessed data
    save_preprocessed_data(
        x_train, x_test, y_train, y_test, config["data"]["preprocessed_data_path"]
    )


if __name__ == "__main__":
    main("config.yml")
