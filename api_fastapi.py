from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Mall Customer Segmentation API")

# Load the trained model
model = joblib.load("mall_customer_segmentation.joblib")

# Input schema
class CustomerInput(BaseModel):
    gender: int        
    annual_income: float
    spending_score: float

@app.get("/")
def root():
    return {"message": "Welcome to Mall Customer Segmentation API"}

@app.post("/predict")
def get_cluster(input: CustomerInput):
    X = pd.DataFrame([[input.gender, input.annual_income, input.spending_score]],columns=["Gender", "Annual Income (k$)", "Spending Score (1-100)"])

    cluster = model.predict(X)[0]
    return {"cluster": int(cluster)}
