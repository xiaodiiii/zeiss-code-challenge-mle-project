# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import boto3
import os
import tempfile
import src.utils.load_config as load_config

def load_data(file_path, using_s3=False, config_path="config.yml"):
    """Load the dataset from a CSV file, either from S3 or local file system."""
    if using_s3:
        # Load configuration
        config = load_config(config_path)
        s3_bucket = config["data"]['s3_bucket']
        s3_key = config["data"]["s3_key"]

        if not s3_bucket or not s3_key:
            raise ValueError("S3 bucket or key not found in configuration file.")

        # Initialize S3 client
        s3_client = boto3.client('s3')

        try:
            # Download the file from S3 to a local temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                s3_client.download_fileobj(s3_bucket, s3_key, tmp_file)
                tmp_file_path = tmp_file.name

            # Load data into DataFrame
            df = pd.read_csv(tmp_file_path, sep=",")
            
            # Clean up temporary file
            os.remove(tmp_file_path)
        
        except Exception as e:
            raise RuntimeError(f"An error occurred while accessing S3: {str(e)}")
    
    else:
        # Load data from local file
        df = pd.read_csv(file_path, sep=",")
    
    return df
