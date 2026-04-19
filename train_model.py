import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = "data/csv/weather_dataset.csv"
MODEL_PATH = "data/skl/weather_model.pkl"
ENCODER_PATH = "data/skl/label_encoder.pkl"

# 1. ฟังก์ชันเตรียมข้อมูล (Preprocessing)
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        print("Error: CSV data file not found.")
        return None, None
    
    df = pd.read_csv(file_path)
    
    # แปลงข้อมูลสถานะอากาศ (String) ให้เป็นตัวเลข (Integer)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['weather'])
    
    return df, le

# 2. ฟังก์ชันจัดการข้อมูลอนุกรมเวลา (Time-Shifting)
def create_time_series_data(df, shift_steps=-3):
    """
    shift_steps: จำนวนแถวที่ต้องการพยากรณ์ล่วงหน้า 
    (ค่าลบหมายถึงการดึงข้อมูลในอนาคตขึ้นมาเป็นเป้าหมาย)
    """
    # เลื่อน Label ของอนาคตมาคู่กับ Feature ของปัจจุบัน
    df['target'] = df['label'].shift(shift_steps)
    
    # กำจัดแถวที่มีค่าว่าง (NaN) จากการเลื่อนข้อมูล
    df_clean = df.dropna().copy()
    
    # กำหนด Input Features และ Target
    X = df_clean[["temp", "humidity", "pressure", "wind_speed"]]
    y = df_clean["target"]
    
    return X, y

# 3. ฟังก์ชันการเทรนโมเดล (Training Process)
def train_model(X, y):
    print(f"Starting training with {len(X)} samples...")
    
    # ใช้ RandomForestClassifier สำหรับการจำแนกประเภทสภาพอากาศ
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# 4. ฟังก์ชันหลักในการรัน Pipeline ทั้งหมด
def run_training_pipeline():
    # ขั้นตอนที่ 1: โหลดและแปลงข้อมูล
    df, le = load_and_preprocess(DATA_FILE)
    if df is None:
        return

    # ขั้นตอนที่ 2: จัดการเรื่อง Time Lag สำหรับการพยากรณ์
    # หากดึงข้อมูลทุก 10 นาที การ shift -3 คือการพยากรณ์ล่วงหน้า 30 นาที
    X, y = create_time_series_data(df, shift_steps=-3)
    
    if len(X) < 10:
        print("Warning: Not enough data to train. Please collect more samples.")
        return

    # ขั้นตอนที่ 3: เริ่มการเทรน
    model = train_model(X, y)

    # ขั้นตอนที่ 4: บันทึกโมเดลและตัวแปลงค่า
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    print("Training successful. Model and Encoder saved.")

# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":
    run_training_pipeline()