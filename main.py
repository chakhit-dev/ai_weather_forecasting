from fastapi import FastAPI
import joblib
import pandas as pd
import requests
import os

app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = ""
CITY = "Bangkok"

MODEL_PATH = "data/skl/weather_model_latest.pkl"
ENCODER_PATH = "data/skl/label_encoder_20260417_054702.pkl"

# -----------------------------
# LOAD MODEL ON START
# -----------------------------
model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# -----------------------------
# GET WEATHER
# -----------------------------
def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    r = requests.get(url)
    return r.json()

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(weather_data):
    df = pd.DataFrame([[
        weather_data["main"]["temp"],
        weather_data["main"]["humidity"],
        weather_data["main"]["pressure"],
        weather_data["wind"]["speed"]
    ]], columns=["temp", "humidity", "pressure", "wind_speed"])

    pred = model.predict(df)[0]
    result = le.inverse_transform([int(pred)])[0]

    return result

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.get("/predict")
def weather_predict():
    weather_data = get_weather()
    result = predict(weather_data)

    return {
        "city": CITY,
        "temp": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "sky": weather_data["weather"][0]["main"],
        "ai_prediction": result
    }