from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import joblib
import numpy as np
import pandas as pd
from typing import List
import logging
import xgboost as xgb
import os
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
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
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
model_path = os.path.normpath(os.path.join(script_dir,"bin/model.xgb"))
model = xgb.Booster()
model.load_model(model_path)
# Load transformer (same one used in training)
transformer_path = os.path.normpath(os.path.join(script_dir,"bin/column_transformer.pkl"))
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
    input_dict = features.dict()
    input_df = pd.DataFrame([input_dict])
    transformed_input = transformer.transform(input_df)
    dmatrix_input = xgb.DMatrix(transformed_input)
    prediction = model.predict(dmatrix_input)
    logging.info(f"Prediction result: {prediction} with the input {input_df}")
    return {"prediction": int(prediction[0])}
