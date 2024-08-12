# -*- coding: utf-8 -*-
"""Entry point to the fastapi app."""
import logging
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from src.main.visualize import plot_dataset, plot_classifier_output, plot_inference_output
from src.main.predict import main as predict_main
from src.main.prepare_data import main as prepare_data_main
from src.main.train import main as train_main
from src.utils.load_data import load_data
from src.utils.load_config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class ConfigPath(BaseModel):
    """A class to represent the configuration path.

    This class inherits from BaseModel and is used to
    store the path to a configuration file."""

    config_path: str


@app.post("/prepare-data/")
async def prepare_data(config: ConfigPath):
    """
    Prepare the data for model training.

    This endpoint triggers the data preparation process
    using the provided configuration path.
    It logs the process and handles any errors
    that may occur during execution.

    Parameters:
    -----------
    config : ConfigPath
        An instance of ConfigPath containing
        the path to the configuration file.

    Returns:
    --------
    dict
        A dictionary with a message indicating the
        success of the data preparation process.

    Raises:
    -------
    HTTPException
        If an error occurs during the data preparation,
        an HTTP 500 error is raised with a detailed message.
    """
    try:
        logger.info(f"Running prepare_data with config: {config.config_path}")
        prepare_data_main(config.config_path)
        return {"message": "Data preparation completed successfully"}
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in data preparation: {str(e)}"
        )


@app.post("/train/")
async def train_model(config: ConfigPath):
    """
    Train the machine learning model.

    This endpoint triggers the model training process
    using the provided configuration path.
    It logs the process and handles any
    errors that may occur during execution.

    Parameters:
    -----------
    config : ConfigPath
        An instance of ConfigPath containing
        the path to the configuration file.

    Returns:
    --------
    dict
        A dictionary with a message indicating
        the success of the model training process.

    Raises:
    -------
    HTTPException
        If an error occurs during model training,
        an HTTP 500 error is raised with a detailed message.
    """
    try:
        logger.info(f"Running train with config: {config.config_path}")
        train_main(config.config_path)
        return {"message": "Model training completed successfully"}
    except Exception as e:
        logger.error(f"Error in train: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in model training: {str(e)}"
        )


@app.post("/predict/")
async def predict(config: ConfigPath):
    """
    Make predictions using the trained model.

    This endpoint triggers the prediction process
    using the provided configuration path.
    It logs the process and handles any
    errors that may occur during execution.

    Parameters:
    -----------
    config : ConfigPath
        An instance of ConfigPath containing
        the path to the configuration file.

    Returns:
    --------
    dict
        A dictionary with a message indicating
        the success of the prediction process.

    Raises:
    -------
    HTTPException
        If an error occurs during the prediction,
        an HTTP 500 error is raised with a detailed message.
    """
    try:
        logger.info(f"Running predict with config: {config.config_path}")
        predict_main(config.config_path)
        return {"message": "Prediction completed successfully"}
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")


@app.get("/plot-dataset/")
async def plot_dataset_endpoint(config_path: str = Query(..., description="Path to the configuration file")):
    """
    Generate and return a plot of the dataset.

    This endpoint generates a plot of the dataset using the provided configuration path.
    It logs the process and handles any errors that may occur during execution.

    Parameters:
    -----------
    config_path : str
        The path to the configuration file.

    Returns:
    --------
    StreamingResponse
        A StreamingResponse object containing the generated plot as a PNG image.

    Raises:
    -------
    HTTPException
        If an error occurs during the plot generation, an HTTP 500 error is raised with a detailed message.
    """
    try:
        logger.info(f"Generating dataset plot with config: {config_path}")
        config_data = load_config(config_path)
        df = load_data(config_data["data"]["raw_data_path"])
        X = df[config_data["features"]].values
        y = df[config_data["target"]].values
        plot_buf = plot_dataset(X, y)
        return StreamingResponse(plot_buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error in generating dataset plot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in generating dataset plot: {str(e)}")

@app.get("/plot-classifier-output/")
async def plot_classifier_output_endpoint(config_path: str = Query(..., description="Path to the configuration file")):
    """
    Generate and return a plot of the classifier output with decision boundaries.

    This endpoint generates a plot of the classifier output using the provided configuration path.
    It logs the process and handles any errors that may occur during execution.

    Parameters:
    -----------
    config_path : str
        The path to the configuration file.

    Returns:
    --------
    StreamingResponse
        A StreamingResponse object containing the generated plot as a PNG image.

    Raises:
    -------
    HTTPException
        If an error occurs during the plot generation, an HTTP 500 error is raised with a detailed message.
    """
    try:
        logger.info(f"Generating classifier output plot with config: {config_path}")
        config_data = load_config(config_path)
        df = load_data(config_data["data"]["raw_data_path"])
        X = df[config_data["features"]].values
        y = df[config_data["target"]].values
        
        # Load pre-trained model
        clf = joblib.load(config_data["model"]["model_output_path"])

        # Calculate accuracy score for the plot
        score = clf.score(X, y)

        plot_buf = plot_classifier_output(X, y, clf, score)
        return StreamingResponse(plot_buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error in generating classifier output plot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in generating classifier output plot: {str(e)}")

@app.get("/plot-inference-output/")
async def plot_inference_output_endpoint(config_path: str = Query(..., description="Path to the configuration file")):
    """
    Generate and return a plot of the inference output with decision boundaries.

    This endpoint generates a plot of the inference output using the provided configuration path.
    It logs the process and handles any errors that may occur during execution.

    Parameters:
    -----------
    config_path : str
        The path to the configuration file.

    Returns:
    --------
    StreamingResponse
        A StreamingResponse object containing the generated plot as a PNG image.

    Raises:
    -------
    HTTPException
        If an error occurs during the plot generation, an HTTP 500 error is raised with a detailed message.
    """
    try:
        logger.info(f"Generating inference output plot with config: {config_path}")
        config_data = load_config(config_path)
        inference_df = load_data(config_data["data"]["inference_data_path"])
        X_inference = inference_df[config_data["features"]].values
        
        # Load pre-trained model
        clf = joblib.load(config_data["model"]["model_output_path"])

        # Make predictions for plotting
        y_pred = clf.predict(X_inference)

        plot_buf = plot_inference_output(X_inference, y_pred, clf)
        return StreamingResponse(plot_buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error in generating inference output plot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in generating inference output plot: {str(e)}")