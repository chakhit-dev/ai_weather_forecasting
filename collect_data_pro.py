import requests
import pandas as pd
import time
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = ""
CITY = "Bangkok"

MODEL_DIR = "data/skl"
DATA_DIR = "data/csv"

MODEL_FILE = os.path.join(MODEL_DIR, "weather_model.pkl")
DATA_FILE = os.path.join(DATA_DIR, "weather_dataset.csv")

FETCH_INTERVAL = 600


# -----------------------------
# INIT FOLDERS
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------
# GET WEATHER DATA
# -----------------------------
def get_weather_data():
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={CITY}&appid={API_KEY}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return None

        data = response.json()

        return {
            "timestamp": time.ctime(),
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "weather": data["weather"][0]["main"],
        }

    except Exception as e:
        print(f"Request failed: {e}")
        return None


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_initial_model():
    print("Training model...")

    data = {
        "temp": [30, 25, 20, 35, 28, 22, 18, 32, 26, 21] * 10,
        "humidity": [80, 85, 90, 40, 75, 95, 30, 70, 82, 88] * 10,
        "pressure": [1010, 1005, 1008, 1012, 1003, 1007, 1015, 1009, 1011, 1006] * 10,
        "wind_speed": [3, 5, 2, 6, 4, 7, 1, 5, 3, 2] * 10,
        "label": [1, 1, 1, 0, 1, 1, 0, 0, 1, 1] * 10,
    }

    df = pd.DataFrame(data)

    X = df[["temp", "humidity", "pressure", "wind_speed"]]
    y = df["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)

    print("Model saved")


# -----------------------------
# SAVE CSV
# -----------------------------
def save_data_to_csv(info):
    df = pd.DataFrame([info])

    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False)
    else:
        df.to_csv(DATA_FILE, mode="a", index=False, header=False)


# -----------------------------
# COUNTDOWN PROGRESS BAR
# -----------------------------
def countdown(seconds):
    for _ in tqdm(range(seconds), desc="Waiting next update", ncols=100):
        time.sleep(1)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    if not os.path.exists(MODEL_FILE):
        train_initial_model()

    print(f"Weather AI System: {CITY}")
    print(f"Update every {FETCH_INTERVAL} seconds\n")

    model = joblib.load(MODEL_FILE)

    while True:
        try:
            current_data = get_weather_data()

            if current_data:
                save_data_to_csv(current_data)

                input_features = pd.DataFrame(
                    [[
                        current_data["temp"],
                        current_data["humidity"],
                        current_data["pressure"],
                        current_data["wind_speed"],
                    ]],
                    columns=["temp", "humidity", "pressure", "wind_speed"],
                )

                prediction = model.predict(input_features)[0]

                print(f"\n{current_data['timestamp']}")
                print("------------------------")
                print(f"Temp: {current_data['temp']} C")
                print(f"Humidity: {current_data['humidity']} %")
                print(f"Wind: {current_data['wind_speed']} m/s")
                print(f"Weather: {current_data['weather']}")

                result = "Rain likely" if prediction == 1 else "Clear sky"
                print(f"AI Prediction: {result}")
                print(f"Saved: {DATA_FILE}")
                print("------------------------")

            else:
                print("Failed to fetch data")

        except KeyboardInterrupt:
            print("\nStopped by user")
            break

        except Exception as e:
            print(f"Error: {e}")

        countdown(FETCH_INTERVAL)