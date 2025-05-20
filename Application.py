from fastapi import FastAPI, HTTPException
from  pydantic import BaseModel

app = FastAPI() 

class features(BaseModel):
    pass 


@app.get("/")
def root():
    return {"message":"welcome to churn prediction API :)"}

@app.get("/health")
def get_health():
    return {"status": "healthy"}     

@app.get("/predict")
def get_prediction(x: features):
    return {"prediction":"prediction-value"}
