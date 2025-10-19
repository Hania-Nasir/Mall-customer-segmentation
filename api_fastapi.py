from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app=FastAPI(title="Mall Customer Segmentation API")

model=joblib.load("mall_customer_segmentation.joblib")

class CustomerInput(BaseModel):
    annual_income:float
    spending_score:float

@app.get("/")
def root():
    return{"message : welcome to mall customer segmentation API"}

@app.post("/Predict")
def get_cluster(annual_income:float,spending_score:float):
    X=pd.DataFrame([[annual_income, spending_score]], columns=["Annual Income (k$)", "Spending Score (1-100)"])
    cluster =model.predict(X)[0]
    return{"cluster":int(cluster)}