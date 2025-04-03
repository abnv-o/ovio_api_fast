from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
print("Current Working Directory:", os.getcwd())
print("Files in Directory:", os.listdir())


app = FastAPI()

# Load trained model
try:
    model = joblib.load("pcos_self_assess_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define request model
class PCOSInput(BaseModel):
    Age: int
    Weight: float
    Height: float
    BMI: float
    Cycle_length: int
    Cycle_regularity: int
    Weight_gain: str
    Hair_growth: str
    Skin_darkening: str
    Pimples: str
    Fast_food: int
    Exercise: int

# Preprocessing function
def preprocess_input(data: PCOSInput):
    try:
        features = [
            "Age", "Weight", "Height", "BMI",
            "Cycle_length", "Cycle_regularity",
            "Weight_gain", "Hair_growth",
            "Skin_darkening", "Pimples",
            "Fast_food", "Exercise"
        ]

        binary_features = ["Weight_gain", "Hair_growth", "Skin_darkening", "Pimples"]
        processed = []

        for feature in features:
            val = getattr(data, feature)
            if feature in binary_features:
                processed.append(1 if val.lower() in ["yes", "y", "1"] else 0)
            else:
                processed.append(float(val))

        return pd.DataFrame([dict(zip(features, processed))])

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(data: PCOSInput):
    df = preprocess_input(data)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": "Risk of PCOS" if prediction == 1 else "No Risk of PCOS",
        "confidence": round(probability, 2)
    }

@app.get("/")
def home():
    return {"message": "PCOS Prediction API is running"}
