"""
Flash Flood / Landslide Risk Prediction — Uttarakhand Full-Scale (v4)
Target: 1,00,000+ rows across extended Uttarakhand districts (2005-2024)
"""

import time
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. EXTENDED LOCATIONS (~40 points covering major Uttarakhand hill districts)
# ---------------------------------------------------------------------------
LOCATIONS = pd.DataFrame([
    # Rudraprayag
    {"name": "Rudraprayag Town", "district": "Rudraprayag", "lat": 30.284, "lon": 78.981},
    {"name": "Kedarnath",        "district": "Rudraprayag", "lat": 30.7346, "lon": 79.0669},
    {"name": "Gaurikund",        "district": "Rudraprayag", "lat": 30.689, "lon": 79.033},
    {"name": "Guptkashi",        "district": "Rudraprayag", "lat": 30.533, "lon": 79.077},
    {"name": "Ukhimath",         "district": "Rudraprayag", "lat": 30.522, "lon": 79.112},
    # Chamoli
    {"name": "Chamoli Town",     "district": "Chamoli", "lat": 30.408, "lon": 79.320},
    {"name": "Joshimath",        "district": "Chamoli", "lat": 30.556, "lon": 79.565},
    {"name": "Badrinath",        "district": "Chamoli", "lat": 30.744, "lon": 79.494},
    {"name": "Karnaprayag",      "district": "Chamoli", "lat": 30.265, "lon": 79.216},
    {"name": "Gopeshwar",        "district": "Chamoli", "lat": 30.439, "lon": 79.330},
    {"name": "Pipalkoti",        "district": "Chamoli", "lat": 30.478, "lon": 79.423},
    {"name": "Govindghat",       "district": "Chamoli", "lat": 30.635, "lon": 79.558},
    # Tehri Garhwal
    {"name": "New Tehri",        "district": "Tehri", "lat": 30.388, "lon": 78.480},
    {"name": "Devprayag",        "district": "Tehri", "lat": 30.146, "lon": 78.598},
    {"name": "Chamba",           "district": "Tehri", "lat": 30.324, "lon": 78.418},
    {"name": "Narendranagar",    "district": "Tehri", "lat": 30.170, "lon": 78.294},
    {"name": "Kirtinagar",       "district": "Tehri", "lat": 30.245, "lon": 78.758},
    {"name": "Pratapnagar",      "district": "Tehri", "lat": 30.452, "lon": 78.361},
    # Pauri Garhwal
    {"name": "Pauri Town",       "district": "Pauri", "lat": 30.149, "lon": 78.777},
    {"name": "Srinagar (UK)",    "district": "Pauri", "lat": 30.222, "lon": 78.784},
    {"name": "Lansdowne",        "district": "Pauri", "lat": 29.836, "lon": 78.686},
    {"name": "Kotdwar",          "district": "Pauri", "lat": 29.747, "lon": 78.526},
    {"name": "Dugadda",          "district": "Pauri", "lat": 29.812, "lon": 78.584},
    {"name": "Satpuli",          "district": "Pauri", "lat": 29.932, "lon": 78.736},
    # Uttarkashi
    {"name": "Uttarkashi Town",  "district": "Uttarkashi", "lat": 30.729, "lon": 78.445},
    {"name": "Gangotri",         "district": "Uttarkashi", "lat": 30.994, "lon": 78.940},
    {"name": "Bhatwari",         "district": "Uttarkashi", "lat": 30.789, "lon": 78.541},
    {"name": "Purola",           "district": "Uttarkashi", "lat": 30.867, "lon": 78.049},
    {"name": "Barkot",           "district": "Uttarkashi", "lat": 30.814, "lon": 78.211},
    {"name": "Yamunotri",        "district": "Uttarkashi", "lat": 30.967, "lon": 78.450},
    {"name": "Harsil",           "district": "Uttarkashi", "lat": 31.033, "lon": 78.733},
    # Pithoragarh & Bageshwar
    {"name": "Pithoragarh Town", "district": "Pithoragarh", "lat": 29.582, "lon": 80.218},
    {"name": "Dharchula",        "district": "Pithoragarh", "lat": 29.851, "lon": 80.528},
    {"name": "Munsiyari",        "district": "Pithoragarh", "lat": 30.067, "lon": 80.245},
    {"name": "Bageshwar Town",   "district": "Bageshwar", "lat": 29.839, "lon": 79.776},
    {"name": "Kausani",          "district": "Bageshwar", "lat": 29.852, "lon": 79.600},
    # Nainital & Champawat
    {"name": "Nainital Town",    "district": "Nainital", "lat": 29.380, "lon": 79.463},
    {"name": "Haldwani",         "district": "Nainital", "lat": 29.218, "lon": 79.513},
    {"name": "Ramnagar",         "district": "Nainital", "lat": 29.392, "lon": 79.126},
    {"name": "Champawat Town",   "district": "Champawat", "lat": 29.336, "lon": 80.091},
    {"name": "Lohaghat",         "district": "Champawat", "lat": 29.408, "lon": 80.088},
]).reset_index(drop=True)
LOCATIONS["location_id"] = ["L" + str(i + 1).zfill(2) for i in range(len(LOCATIONS))]

# ---------------------------------------------------------------------------
# 2. ELEVATION + SLOPE 
# ---------------------------------------------------------------------------
def fetch_elevation(lat: float, lon: float, retries: int = 3) -> float:
    url = "https://api.open-meteo.com/v1/elevation"
    for _ in range(retries):
        try:
            resp = requests.get(url, params={"latitude": lat, "longitude": lon}, timeout=20)
            resp.raise_for_status()
            return resp.json()["elevation"][0]
        except requests.RequestException:
            time.sleep(2)
    return 1500.0

def compute_slope_deg(lat: float, lon: float, offset_deg: float = 0.005) -> float:
    elev_here = fetch_elevation(lat, lon)
    elev_offset = fetch_elevation(lat + offset_deg, lon)
    horizontal_distance_m = offset_deg * 111_000
    rise_m = abs(elev_offset - elev_here)
    return float(np.degrees(np.arctan(rise_m / horizontal_distance_m)))

def build_location_metadata(locations: pd.DataFrame) -> pd.DataFrame:
    elevations, slopes = [], []
    print("Fetching elevation and slope profiles across Uttarakhand grid...")
    for _, loc in locations.iterrows():
        elev = fetch_elevation(loc["lat"], loc["lon"])
        slope = compute_slope_deg(loc["lat"], loc["lon"])
        elevations.append(elev)
        slopes.append(slope)
        time.sleep(0.2)
    df = locations.copy()
    df["elevation_m"] = elevations
    df["slope_deg"] = slopes
    return df


# ---------------------------------------------------------------------------
# 3. HISTORICAL WEATHER ARCHIVE (2005 - 2024) - Updated with better retries
# ---------------------------------------------------------------------------
def fetch_historical_weather(locations: pd.DataFrame, start_year: int = 2005,
                              end_year: int = 2024, monsoon_months=(5, 6, 7, 8, 9, 10)) -> pd.DataFrame:
    all_rows = []
    print(f"Fetching historical weather archive ({start_year}-{end_year}) for {len(locations)} stations...")
    
    for _, loc in locations.iterrows():
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": loc["lat"], "longitude": loc["lon"],
            "start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31",
            "hourly": "precipitation,soil_moisture_0_to_1cm", "timezone": "Asia/Kolkata",
        }
        
        success = False
        hourly = None
        # 4 retries ke sath loop chalega
        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=120)
                if resp.status_code == 429:  # Rate limit hit
                    print(f"  Rate limit hit for {loc['name']}. Sleeping for 10s...")
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                hourly = resp.json().get("hourly")
                if hourly:
                    success = True
                    break
            except requests.RequestException:
                time.sleep(3 * (attempt + 1)) # Progressive backoff: 3s, 6s, 9s...

        if not success or not hourly:
            print(f"  Skipping {loc['name']} due to persistent network timeout.")
            continue

        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            "rainfall_mm": hourly["precipitation"],
            "soil_moisture_frac": hourly["soil_moisture_0_to_1cm"],
        })
        df["date"] = df["datetime"].dt.floor("D")
        daily = df.groupby("date").agg(
            rainfall_mm=("rainfall_mm", "sum"),
            soil_moisture_frac=("soil_moisture_frac", "mean"),
        ).reset_index()

        daily = daily[daily["date"].dt.month.isin(monsoon_months)].copy()
        daily["location_id"] = loc["location_id"]
        daily["name"] = loc["name"]
        daily["district"] = loc["district"]
        daily["lat"] = loc["lat"]
        daily["lon"] = loc["lon"]
        daily["elevation_m"] = loc["elevation_m"]
        daily["slope_deg"] = loc["slope_deg"]
        daily["soil_moisture"] = (daily["soil_moisture_frac"].fillna(0) * 100).clip(0, 100)
        daily = daily.drop(columns=["soil_moisture_frac"])

        all_rows.append(daily)
        print(f"  Successfully fetched: {loc['name']}")
        time.sleep(1.0) # Har request ke baad 1 second ka gap taaki connection stable rahe

    if not all_rows:
        raise ValueError("All station requests failed. Please check your internet connection.")

    data = pd.concat(all_rows, ignore_index=True)
    return data.sort_values(["location_id", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. OFFLINE INCIDENT LABELS (NASA CSV) & 30KM MATCHING
# ---------------------------------------------------------------------------
UK_BBOX = {"xmin": 77.8, "ymin": 29.0, "xmax": 81.0, "ymax": 31.3}

def fetch_coolr_events(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    try:
        df = pd.read_csv("nasa_landslides.csv", low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame(columns=["lat", "lon", "date"])

    col_mapping = {"event_date": "date", "date": "date", "latitude": "lat", "lat": "lat", "longitude": "lon", "lon": "lon"}
    df = df.rename(columns=lambda x: col_mapping.get(x.lower(), x))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "date"])

    mask_bbox = (
        (df["lon"] >= UK_BBOX["xmin"]) & (df["lon"] <= UK_BBOX["xmax"]) &
        (df["lat"] >= UK_BBOX["ymin"]) & (df["lat"] <= UK_BBOX["ymax"])
    )
    df = df[mask_bbox]
    df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]
    
    events = df[["lat", "lon", "date"]].copy()
    events["date"] = events["date"].dt.floor("D")
    return events

def match_events_to_locations(events: pd.DataFrame, locations: pd.DataFrame, radius_km: float = 30.0) -> set:
    matched = set()
    if events.empty:
        return matched
    for _, ev in events.iterrows():
        dlat = np.radians(locations["lat"] - ev["lat"])
        dlon = np.radians(locations["lon"] - ev["lon"])
        a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(ev["lat"])) * np.cos(np.radians(locations["lat"])) * np.sin(dlon / 2) ** 2)
        dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))
        nearest_idx = dist_km.idxmin()
        if dist_km[nearest_idx] <= radius_km:
            matched.add((locations.loc[nearest_idx, "location_id"], ev["date"]))
    return matched

# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_features(weather: pd.DataFrame, matched_events: set) -> pd.DataFrame:
    data = weather.sort_values(["location_id", "date"]).copy()
    data["rain_24hr"] = data.groupby("location_id")["rainfall_mm"].transform(lambda s: s.rolling(1, min_periods=1).sum())
    data["rain_72hr"] = data.groupby("location_id")["rainfall_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    data["rain_rate_change"] = data.groupby("location_id")["rainfall_mm"].diff().fillna(0)

    data["flood_event"] = data.apply(lambda r: 1 if (r["location_id"], r["date"]) in matched_events else 0, axis=1)

    event_dates_by_loc = {}
    for loc_id, ev_date in matched_events:
        event_dates_by_loc.setdefault(loc_id, []).append(ev_date)

    def rolling_incident_count(row):
        loc_events = event_dates_by_loc.get(row["location_id"], [])
        window_start = row["date"] - pd.Timedelta(days=5 * 365)
        return sum(1 for d in loc_events if window_start <= d < row["date"])

    data["historical_incidents"] = data.apply(rolling_incident_count, axis=1)
    return data

# ---------------------------------------------------------------------------
# 6. MODEL TRAINING & EVALUATION (Threshold = 0.45)
# ---------------------------------------------------------------------------
FEATURES = [
    "elevation_m", "slope_deg", "historical_incidents",
    "rainfall_mm", "rain_24hr", "rain_72hr", "rain_rate_change", "soil_moisture",
]

def train_model(data: pd.DataFrame, threshold: float = 0.45):
    X = data[FEATURES]
    y = data["flood_event"]

    print(f"\nTotal Dataset Size: {len(data)} rows | Positive Disasters: {y.sum()} ({100 * y.mean():.3f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() >= 2 else None
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\n--- Classification Report (Threshold = {threshold}) ---")
    print(classification_report(y_test, y_pred, labels=[0, 1],
                                 target_names=["No Event", "Flood/Landslide Risk"], zero_division=0))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(f"Total Test Cases Evaluated: {len(y_test)}")

    return model

# ---------------------------------------------------------------------------
# EXECUTION FLOW
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Prepare data grid, fetch historical archive & train model
    locations = build_location_metadata(LOCATIONS)
    weather = fetch_historical_weather(locations, start_year=2005, end_year=2024)
    events = fetch_coolr_events(start_year=2000, end_year=2024)
    matched = match_events_to_locations(events, locations, radius_km=30.0)
    print(f"Matched {len(matched)} total disaster events across all Uttarakhand regions.")
    
    feature_data = build_features(weather, matched)
    feature_data.to_csv("flood_training_data_100k.csv", index=False)

    model = train_model(feature_data, threshold=0.45)
    joblib.dump(model, "flood_model_uk.pkl")
    print("\nModel successfully trained and saved as flood_model_uk.pkl!")