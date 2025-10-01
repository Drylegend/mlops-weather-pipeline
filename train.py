# train.py

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

from meteostat import Point, Hourly
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def fetch_data():
    """Fetches the last year of historical data from Meteostat."""
    print("--- Fetching historical data ---")
    end = datetime.now()
    start = end - timedelta(days=365)
    bengaluru = Point(12.9716, 77.5946)
    data = Hourly(bengaluru, start, end)
    df = data.fetch()
    print(f"Successfully downloaded {df.shape[0]} hours of data.")
    return df

def create_api_features(df):
    """Selects and cleans the features needed for the API model."""
    print("--- Engineering API-ready features ---")
    api_features_map = {
        'temp': 'temperature', 'rhum': 'humidity', 'prcp': 'precipitation',
        'wspd': 'wind_speed', 'pres': 'pressure'
    }
    df_api = df[list(api_features_map.keys())].rename(columns=api_features_map)
    df_api.ffill(inplace=True)
    df_api['hour'] = df_api.index.hour
    df_api['day_of_week'] = df_api.index.dayofweek
    df_api['month'] = df_api.index.month
    df_api.dropna(inplace=True)
    return df_api

def train_and_save_artifacts(df):
    """Trains and saves the final API model and its artifacts."""
    print("--- Starting training and artifact saving ---")
    
    # 1. Create Target and Features
    df_copy = df.copy()
    df_copy['will_rain'] = (df_copy['precipitation'] > 0).astype(int)
    y = df_copy['will_rain']
    X = df_copy.drop(columns=['precipitation', 'will_rain'])
    model_columns = X.columns.tolist()

    # 2. Split Data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Fit the Scaler on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Resample the training data
    n_minority_samples = y_train.value_counts().get(1, 0)
    if n_minority_samples > 1:
        k = min(n_minority_samples - 1, 5)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    # 5. Train the Final Model
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    print("Final API model trained.")

    # 6. Save the Artifacts
    joblib.dump(model, 'api_model.joblib')
    joblib.dump(scaler, 'api_scaler.joblib')
    with open('api_columns.json', 'w') as f:
        json.dump(model_columns, f)
        
    print("\nAPI artifacts saved successfully!")

# --- Main execution block ---
if __name__ == "__main__":
    raw_data = fetch_data()
    api_ready_data = create_api_features(raw_data)
    train_and_save_artifacts(api_ready_data)