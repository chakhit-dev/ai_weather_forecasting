import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = "data/csv/weather_dataset_test.csv"
MODEL_DIR = "data/skl"

# -----------------------------
# 1. Load + Preprocess
# -----------------------------
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        print("Error: CSV data file not found.")
        return None, None

    df = pd.read_csv(file_path)

    # Encode label
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['weather'])

    return df, le

# -----------------------------
# 2. Time Series Shift
# -----------------------------
def create_time_series_data(df, shift_steps=-3):
    """
    shift_steps: จำนวน step ที่ต้องการพยากรณ์ล่วงหน้า
    """
    df = df.copy()

    # สร้าง target แบบ future shift
    df['target'] = df['label'].shift(shift_steps)

    # ลบ NaN
    df_clean = df.dropna().copy()

    # Features
    X = df_clean[["temp", "humidity", "pressure", "wind_speed"]]
    y = df_clean["target"]

    return X, y

# -----------------------------
# 3. Train Model
# -----------------------------
def train_model(X, y):
    print(f"Starting training with {len(X)} samples...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)
    return model

# -----------------------------
# 4. Save Model (VERSIONED)
# -----------------------------
def save_artifacts(model, encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(MODEL_DIR, f"weather_model_{timestamp}.pkl")
    encoder_path = os.path.join(MODEL_DIR, f"label_encoder_{timestamp}.pkl")

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    # optional: บันทึก latest ไว้ใช้งานง่าย
    joblib.dump(model, os.path.join(MODEL_DIR, "weather_model_latest.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder_latest.pkl"))

    print("Model saved:")
    print(" -", model_path)
    print(" -", encoder_path)
    print("Latest model updated.")

# -----------------------------
# 5. Pipeline
# -----------------------------
def run_training_pipeline():
    df, le = load_and_preprocess(DATA_FILE)
    if df is None:
        return

    X, y = create_time_series_data(df, shift_steps=-3)

    if len(X) < 10:
        print("Warning: Not enough data to train. Please collect more samples.")
        return

    model = train_model(X, y)

    save_artifacts(model, le)

    print("Training completed successfully.")

# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":
    run_training_pipeline()