import requests
import joblib
import pandas as pd
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
API_KEY = ""
CITY = "Bangkok"
MODEL_PATH = "data/skl/weather_model_latest.pkl"
ENCODER_PATH = "data/skl/label_encoder_20260417_054702.pkl"

# 1. ฟังก์ชันโหลดโมเดลและตัวแปลงค่า
def load_trained_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("Error: Model or Encoder file not found. Please run train_model.py first.")
        return None, None
    
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le

# 2. ฟังก์ชันดึงข้อมูลปัจจุบันจาก API
def get_current_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

# 3. ฟังก์ชันพยากรณ์ผล
def make_prediction(model, le, weather_data):
    # เตรียม Features ให้เหมือนกับตอนเทรน
    input_features = pd.DataFrame([[
        weather_data["main"]["temp"],
        weather_data["main"]["humidity"],
        weather_data["main"]["pressure"],
        weather_data["wind"]["speed"]
    ]], columns=["temp", "humidity", "pressure", "wind_speed"])

    # ทำนายผล (จะได้ออกมาเป็นตัวเลข)
    prediction_code = model.predict(input_features)[0]

    # แปลงตัวเลขกลับเป็นคำศัพท์ (เช่น 0 -> Clear)
    # ใช้ .astype(int) เพื่อป้องกัน error กรณีค่าที่ได้เป็น float
    prediction_text = le.inverse_transform([int(prediction_code)])[0]
    
    return prediction_text

# 4. ฟังก์ชันหลัก
def run_prediction():
    # โหลดโมเดล
    model, le = load_trained_assets()
    if model is None: return

    # ดึงข้อมูลจริง
    weather_data = get_current_weather()
    if weather_data:
        # พยากรณ์
        result = make_prediction(model, le, weather_data)
        
        print(f"--- Weather Prediction for {CITY} ---")
        print(f"Current Temp: {weather_data['main']['temp']} C")
        print(f"Current Humidity: {weather_data['main']['humidity']} %")
        print(f"Current Sky: {weather_data['weather'][0]['main']}")
        print("---------------------------------------")
        print(f"AI Prediction (Next Period): {result}")
        print("---------------------------------------")

if __name__ == "__main__":
    run_prediction()