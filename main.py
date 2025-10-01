# .\venv\Scripts\activate
#pip install fastapi "uvicorn[standard]" scikit-learn joblib pandas
#uvicorn main:app --reload

# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

# Create a FastAPI app instance
app = FastAPI()

# --- 1. Load Artifacts at Startup ---
# This ensures the model is loaded only once
model = joblib.load('api_model.joblib')
scaler = joblib.load('api_scaler.joblib')
model_columns = json.load(open('api_columns.json'))

# --- 2. Define the Input Data Schema using Pydantic ---
# This defines the structure and data types for an incoming request
class WeatherInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    pressure: float
    hour: int
    day_of_week: int
    month: int

# --- 3. Create the Prediction Endpoint ---
@app.post("/predict")
def predict_rain(data: WeatherInput):
    # Convert the input data into a pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Ensure the columns are in the same order as during training
    input_df = input_df[model_columns]
    
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)
    
    # Make the prediction and get the probability
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] # Probability for class '1' (Rain)
    
    # Define the human-readable result
    result = "It will rain" if prediction == 1 else "It will not rain"
    
    # Return the prediction and probability as a JSON response
    return {
        "prediction": result,
        "probability_of_rain": float(probability)
    }

# --- 4. Create a Root Endpoint ---
# A simple endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Prediction API!"}