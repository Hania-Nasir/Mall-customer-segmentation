from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="Mall Customer Segmentation API")

# Load trained model
model = joblib.load("mall_customer_segmentation.joblib")

# Define input schema
class CustomerInput(BaseModel):
    gender: str             # "male" or "female"
    age: float              # Original age (will be log-transformed)
    income: float           # Original income (will be log-transformed)
    spending_score: float   # Spending score (1–100)

@app.get("/")
def root():
    return {"message": "Welcome to Mall Customer Segmentation API"}

@app.post("/predict")
def get_cluster(input: CustomerInput):
    # Convert gender text to numeric
    gender = input.gender.strip().lower()
    if gender == "male":
        gender_val = 1
    elif gender == "female":
        gender_val = 0
    else:
        raise HTTPException(status_code=400, detail="Invalid gender. Use 'male' or 'female'.")

    # Ensure positive values for log transform
    if input.age <= 0 or input.income <= 0:
        raise HTTPException(status_code=400, detail="Age and income must be positive numbers.")

    # Apply log transformations (same as used during training)
    age_log = np.log(input.age)
    income_log = np.log(input.income)

    # Prepare input in the same order/column names used in training
    X = pd.DataFrame(
        [[gender_val, input.spending_score, age_log, income_log]],
        columns=["Gender", "Spending Score (1-100)", "Age_log", "Income_log"]
    )

    # Make prediction
    try:
        cluster = int(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {"cluster": cluster}
