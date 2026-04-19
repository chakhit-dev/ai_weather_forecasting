import pandas as pd
import os

# 1. ตั้งค่า Path ไฟล์
input_file = "data/weatherAUS.csv" 
output_file = "data/csv/weather_dataset_test.csv"

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
else:
    # โหลดข้อมูล
    df_raw = pd.read_csv(input_file)

    # 2. เลือกและเปลี่ยนชื่อคอลัมน์
    selected_columns = {
        'Temp3pm': 'temp',
        'Humidity3pm': 'humidity',
        'Pressure3pm': 'pressure',
        'WindSpeed3pm': 'wind_speed',
        'RainToday': 'weather'
    }

    # สร้าง DataFrame ใหม่
    df_mapped = df_raw[list(selected_columns.keys())].copy()
    df_mapped.rename(columns=selected_columns, inplace=True)

    # 3. Clean Data: ลบค่าว่าง
    df_mapped.dropna(inplace=True)

    # 4. แปลงค่า Weather ให้ตรงกับ API (No -> Clear, Yes -> Rain)
    # วิธีนี้จะทำให้ LabelEncoder ในไฟล์เทรนทำงานได้ถูกต้อง
    weather_map = {'No': 'Clear', 'Yes': 'Rain'}
    df_mapped['weather'] = df_mapped['weather'].map(weather_map)

    # 5. เติมคอลัมน์ timestamp (เพื่อให้โครงสร้างไฟล์เหมือนกับที่ดึงจาก API)
    df_mapped['timestamp'] = "Historical"

    # จัดเรียงลำดับคอลัมน์ใหม่ให้เหมือนไฟล์ที่ดึงจาก API
    column_order = ['timestamp', 'temp', 'humidity', 'pressure', 'wind_speed', 'weather']
    df_mapped = df_mapped[column_order]

    # 6. บันทึกไฟล์
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_mapped.to_csv(output_file, index=False)

    print(f"Successfully converted {len(df_mapped)} rows to {output_file}")
    print("\n--- Sample Output Data ---")
    print(df_mapped.head())