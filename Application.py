from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()
#pip install fastapi uvicorn add to req.txt
class Features(BaseModel):
    Geography: str = Field(..., description="Country of the customer, e.g., 'France', 'Spain', 'Germany'")
    Gender: str = Field(..., description="Customer's gender: 'Male' or 'Female'")
    CreditScore: int = Field(..., description="Credit score of the customer")
    Age: int = Field(..., description="Age of the customer in years")
    Tenure: int = Field(..., description="Number of years the customer has been with the bank")
    Balance: float = Field(..., description="Account balance of the customer")
    NumOfProducts: int = Field(..., description="Number of bank products the customer is using")
    HasCrCard: int = Field(..., description="Whether the customer has a credit card (0 or 1)")
    IsActiveMember: int = Field(..., description="Whether the customer is an active member (0 or 1)")
    EstimatedSalary: float = Field(..., description="Estimated yearly salary of the customer")



@app.get("/")
def root():
    return {"message":"welcome to churn prediction API :)"}

@app.get("/health")
def get_health():
    return {"status": "healthy"}     

@app.get("/predict")
def get_prediction(x: Features):
    return {"prediction":"prediction-value"}
