from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import joblib
import numpy as np
import pandas as pd
from typing import List
import logging

app = FastAPI()
#remeber requirements include fastapi univcorn logging
# Define input schema
class Features(BaseModel):
    Geography: str
    Gender: str
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Load model from MLflow
model_uri = "models:/top_churn_models/production"
model = mlflow.sklearn.load_model(model_uri)

# Load transformer (same one used in training)
transformer_path = "column_transformer.pkl"  # Ensure this file exists in your project directory
transformer = joblib.load(transformer_path)

#logging setup 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
@app.get("/")
def root():
    logging.info(f"welcoming message")
    return {"message": "Welcome to the churn prediction API!"}

@app.get("/health")
def health_check():
    logging.info(f"checking health")
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: Features):
    # Create DataFrame from input
    input_dict = features.dict()
    input_df = pd.DataFrame([input_dict])

    # Transform input using saved pipeline
    transformed_input = transformer.transform(input_df)

    # Predict
    prediction = model.predict(transformed_input)
    logging.info(f"Prediction result: {result} with the input {input_df}")
    return {"prediction": int(prediction[0])}
