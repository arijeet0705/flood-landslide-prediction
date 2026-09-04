import time
import pandas as pd
import requests
import joblib
from flash_flood_model import LOCATIONS  # Imports locations safely without triggering training

def run_live_dashboard(model_path="flood_model_uk.pkl", threshold=0.45):
    print("==============================================")
    print("      REAL-TIME LIVE SCORING & ALERTS       ")
    print("==============================================")
    
    # 1. Load the pre-trained model
    try:
        model = joblib.load(model_path)
        print("Model 'flood_model_uk.pkl' loaded successfully!\n")
    except FileNotFoundError:
        print("❌ Error: Trained model not found! Pehle flash_flood_model3.py run karke model train kar.")
        return

    # 2. Loop through locations and fetch live weather via Open-Meteo API
    for _, loc in LOCATIONS.iterrows():
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": loc["lat"], 
            "longitude": loc["lon"],
            "current": "precipitation,soil_moisture_0_to_1cm",
            "timezone": "Asia/Kolkata"
        }
        
        success = False
        current = {}
        # Quick retry loop for live forecast
        for _ in range(3):
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                current = resp.json().get("current", {})
                success = True
                break
            except requests.RequestException:
                time.sleep(1)

        if not success:
            print(f"[{loc['district']}] {loc['name']:<18} | ⚠️ Live fetch failed (Network Timeout)")
            continue

        rain = current.get("precipitation", 0.0) or 0.0
        soil_frac = current.get("soil_moisture_0_to_1cm", 0.0) or 0.0
        soil_moisture = float(soil_frac * 100)
        
        # Fallback values if elevation/slope columns are missing in imported LOCATIONS
        elev = loc.get("elevation_m", 1500.0)
        slope = loc.get("slope_deg", 15.0)

        features = pd.DataFrame([{
            "elevation_m": elev,
            "slope_deg": slope,
            "historical_incidents": 2, 
            "rainfall_mm": rain,
            "rain_24hr": rain * 1.5,
            "rain_72hr": rain * 2.0,
            "rain_rate_change": 0.0,
            "soil_moisture": soil_moisture
        }])

        prob = model.predict_proba(features)[0][1]
        
        if prob >= 0.7:
            status = "🔴 HIGH RISK (RED ALERT)"
        elif prob >= threshold:
            status = "🟡 MEDIUM RISK (ORANGE ALERT)"
        else:
            status = "🟢 SAFE (GREEN)"

        print(f"[{loc['district']}] {loc['name']:<18} | Rain: {rain}mm | Soil: {soil_moisture:.1f}% | Prob: {prob:.2f} -> {status}")
        time.sleep(0.1) # Small gap between requests

if __name__ == "__main__":
    run_live_dashboard()