import requests
import pandas as pd
import time
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = ""
CITY = "Bangkok"

MODEL_DIR = "data/skl"
DATA_DIR = "data/csv"

MODEL_FILE = os.path.join(MODEL_DIR, "weather_model.pkl")
DATA_FILE = os.path.join(DATA_DIR, "weather_dataset.csv")


# -----------------------------
# INIT FOLDERS
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------
# 1. GET WEATHER DATA
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
# 2. TRAIN MODEL
# -----------------------------
def train_initial_model():
    print("กำลังเทรนโมเดล AI...")

    data = {
        "temp": [30, 25, 20, 35, 28, 22, 18, 32, 26, 21] * 10,
        "humidity": [80, 85, 90, 40, 75, 95, 30, 70, 82, 88] * 10,
        "pressure": [1010, 1005, 1008, 1012, 1003, 1007, 1015, 1009, 1011, 1006] * 10,
        "wind_speed": [3, 5, 2, 6, 4, 7, 1, 5, 3, 2] * 10,
        "label": [1, 1, 1, 0, 1, 1, 0, 0, 1, 1] * 10,  # 1=Rain, 0=Clear
    }

    df = pd.DataFrame(data)

    X = df[["temp", "humidity", "pressure", "wind_speed"]]
    y = df["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)

    print("บันทึกโมเดลเรียบร้อย!")


# -----------------------------
# 3. SAVE CSV
# -----------------------------
def save_data_to_csv(info):
    df = pd.DataFrame([info])

    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False)
    else:
        df.to_csv(DATA_FILE, mode="a", index=False, header=False)


# -----------------------------
# MAIN
# -----------------------------
# if __name__ == "__main__":

#     if not os.path.exists(MODEL_FILE):
#         train_initial_model()

#     print(f"--- Weather AI System: {CITY} ---")

#     current_data = get_weather_data()

#     if current_data:

#         # save data
#         save_data_to_csv(current_data)
#         print(f"Saved -> {DATA_FILE}")

#         # load model
#         model = joblib.load(MODEL_FILE)

#         input_features = pd.DataFrame(
#             [[
#                 current_data["temp"],
#                 current_data["humidity"],
#                 current_data["pressure"],
#                 current_data["wind_speed"],
#             ]],
#             columns=["temp", "humidity", "pressure", "wind_speed"],
#         )

#         prediction = model.predict(input_features)[0]

#         # output
#         print("\n========================")
#         print(f"อุณหภูมิ: {current_data['temp']} °C")
#         print(f"ความชื้น: {current_data['humidity']} %")
#         print(f"ความกดอากาศ: {current_data['pressure']}")
#         print(f"ลม: {current_data['wind_speed']} m/s")
#         print(f"สภาพอากาศจริง: {current_data['weather']}")
#         print("------------------------")

#         result = "ฝนอาจจะตก 🌧️" if prediction == 1 else "ท้องฟ้าแจ่มใส ☀️"
#         print(f"AI ทำนาย: {result}")
#         print("========================\n")

#     else:
#         print("ไม่สามารถดึงข้อมูลอากาศได้")


# ... (โค้ดส่วนบนเหมือนเดิมจนถึงส่วน MAIN) ...

# -----------------------------
# MAIN (Update with while loop)
# -----------------------------
if __name__ == "__main__":

    # เช็คว่ามีโมเดลหรือยัง ถ้าไม่มีให้เทรนก่อนเริ่ม Loop
    if not os.path.exists(MODEL_FILE):
        train_initial_model()

    # ตั้งค่าความถี่ในการดึงข้อมูล (หน่วยเป็นวินาที)
    # เช่น 3600 = 1 ชั่วโมง, 600 = 10 นาที
    FETCH_INTERVAL = 600 

    print(f"--- Weather AI System: {CITY} (Loop Active) ---")
    print(f"ดึงข้อมูลทุกๆ {FETCH_INTERVAL} วินาที... กด Ctrl+C เพื่อหยุด")

    while True:
        try:
            current_data = get_weather_data()

            if current_data:
                # 1. บันทึกข้อมูล (Data Logging)
                save_data_to_csv(current_data)
                
                # 2. โหลดโมเดลมาพยากรณ์
                # หมายเหตุ: โหลดครั้งเดียวข้างนอก loop ก็ได้เพื่อประหยัดทรัพยากร
                # แต่โหลดในนี้จะดีถ้าคุณมีการอัปเดตไฟล์โมเดลระหว่างรัน
                model = joblib.load(MODEL_FILE)

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

                # 3. แสดงผลลัพธ์
                print(f"\n[{current_data['timestamp']}]")
                print("------------------------")
                print(f"🌡️  Temp: {current_data['temp']}°C | 💧 Humid: {current_data['humidity']}%")
                print(f"🌪️  Wind: {current_data['wind_speed']} m/s | ☁️  Actual: {current_data['weather']}")
                
                result = "ฝนอาจจะตก 🌧️" if prediction == 1 else "ท้องฟ้าแจ่มใส ☀️"
                print(f"🤖 AI Prediction: {result}")
                print(f"💾 Saved to: {DATA_FILE}")
                print("------------------------")
                print(f"Waiting for next update in {FETCH_INTERVAL}s...")

            else:
                print("⚠️ ไม่สามารถดึงข้อมูลได้ จะลองใหม่ในรอบถัดไป")

        except KeyboardInterrupt:
            print("\n🛑 หยุดการทำงานโดยผู้ใช้")
            break
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดไม่คาดคิด: {e}")

        # พักการทำงานตามเวลาที่กำหนด
        time.sleep(FETCH_INTERVAL)