import pytest
from fastapi.testclient import TestClient
from Application import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the churn prediction API!"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    test_data = {
                "Geography": "France",
                "Gender": "Female",
                "CreditScore": 619,
                "Age": 42,
                "Tenure": 2,
                "Balance": 0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348
                }
    
    response = client.get("/predict", params=test_data)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [0, 1] 
