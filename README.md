# MLOps End-to-End Weather Prediction Pipeline

This project demonstrates a full, end-to-end Machine Learning Operations (MLOps) pipeline designed to solve the problem of **model drift** by creating a system that automatically retrains itself.

## 1. Problem Statement
The central challenge this project addresses is **‘model drift,’** a critical issue where the performance of production machine learning models degrades as real-world data patterns evolve. This project engineers a robust solution by designing and deploying a fully automated, end-to-end MLOps pipeline with a scheduled retraining mechanism.

The use case is a practical weather forecast for **Bengaluru**, specifically predicting the probability of rain.

## 2. Key Features
* **Prediction API:** A REST API built with **FastAPI** to serve real-time rain predictions.
* **Automated Training (CI/CD/CT):** A Continuous Training pipeline using **GitHub Actions** that automatically runs a training script on a weekly schedule.
* **Containerized Application:** The entire application is packaged with **Docker**, making it portable and easy to deploy anywhere.
* **Data-Driven Modeling:** The final model is a **Random Forest Classifier** selected after a rigorous process of Exploratory Data Analysis (EDA), feature engineering, and comparison with a baseline model.

## 3. Tech Stack
* **Language:** Python
* **Data Science:** Pandas, Scikit-learn, Imbalanced-learn
* **API:** FastAPI, Uvicorn
* **Deployment:** Docker
* **Automation:** GitHub Actions

## 4. Local Setup and Installation

To run this project on your local machine, follow these steps:

**1. Clone the Repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**2. Create and Activate a Virtual Environment:**
```bash
# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the API Server:**
```bash
uvicorn main:app --reload
```

**5. Access the API:**
Open your browser and go to `http://127.0.0.1:8000/docs` to access the interactive API documentation.

## 5. Usage (API Endpoint)

The API has one main endpoint: `/predict`.

* **Method:** `POST`
* **Body:** A JSON object with the current weather features.
* **Example Request (`curl`):**
    ```bash
    curl -X 'POST' \
      '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "temperature": 22,
      "humidity": 95,
      "wind_speed": 6.5,
      "pressure": 1004,
      "hour": 16,
      "day_of_week": 3,
      "month": 7
    }'
    ```
* **Example Success Response:**
    ```json
    {
      "prediction": "It will rain",
      "probability_of_rain": 0.85
    }
    ```

## 6. The Automated Training Pipeline

This project's core is the automated retraining pipeline defined in `.github/workflows/scheduled-retraining.yml`.

* **Trigger:** The workflow runs automatically every Sunday at midnight UTC.
* **Process:**
    1.  The job checks out the latest code.
    2.  It runs the `train.py` script.
    3.  The script fetches the last 365 days of weather data from the Meteostat API.
    4.  It engineers an advanced feature set (time-based, lagged, etc.).
    5.  It retrains the Random Forest model on this new, fresh data.
    6.  The final step commits the newly trained model artifacts (`api_model.joblib`, `api_scaler.joblib`, etc.) back to the repository.

This automated loop ensures the model is always up-to-date with the latest weather patterns, effectively solving the problem of model drift.