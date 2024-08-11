# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
from src.main.prepare_data import main as prepare_data_main
from src.main.train import main as train_main
from src.main.predict import main as predict_main

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ConfigPath(BaseModel):
    config_path: str

@app.post("/prepare-data/")
async def prepare_data(config: ConfigPath):
    try:
        logger.info(f"Running prepare_data with config: {config.config_path}")
        prepare_data_main(config.config_path)
        return {"message": "Data preparation completed successfully"}
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in data preparation: {str(e)}")

@app.post("/train/")
async def train_model(config: ConfigPath):
    try:
        logger.info(f"Running train with config: {config.config_path}")
        train_main(config.config_path)
        return {"message": "Model training completed successfully"}
    except Exception as e:
        logger.error(f"Error in train: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in model training: {str(e)}")

@app.post("/predict/")
async def predict(config: ConfigPath):
    try:
        logger.info(f"Running predict with config: {config.config_path}")
        predict_main(config.config_path)
        return {"message": "Prediction completed successfully"}
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")
