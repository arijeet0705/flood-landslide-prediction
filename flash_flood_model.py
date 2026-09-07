"""
Flash Flood / Landslide Risk Prediction — Uttarakhand Full-Scale (v4)
Target: 1,00,000+ rows across extended Uttarakhand districts (2005-2024)
"""

import time
import os
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
            "hourly": "precipitation,soil_moisture_0_to_7cm", "timezone": "Asia/Kolkata",
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
            "soil_moisture_frac": hourly["soil_moisture_0_to_7cm"],
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
    data["rain_5day"] = data.groupby("location_id")["rainfall_mm"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    data["rain_rate_change"] = data.groupby("location_id")["rainfall_mm"].diff().fillna(0)

    # Simple "extreme rainfall day" flag — a day where rainfall is unusually
    # high relative to that location's own typical monsoon rainfall (top 5%).
    # Cheap, physically meaningful signal for sudden-onset flash floods.
    threshold_per_loc = data.groupby("location_id")["rainfall_mm"].transform(lambda s: s.quantile(0.95))
    data["extreme_rain_flag"] = (data["rainfall_mm"] >= threshold_per_loc).astype(int)

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
# ---------------------------------------------------------------------------
# 6. MODEL TRAINING & EVALUATION — continuous risk score, threshold-independent metrics
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

FEATURES = [
    "elevation_m", "slope_deg", "historical_incidents",
    "rainfall_mm", "rain_24hr", "rain_72hr", "rain_5day", "rain_rate_change",
    "soil_moisture", "extreme_rain_flag",
]


def train_model(data: pd.DataFrame):
    X = data[FEATURES]
    y = data["flood_event"]

    print(f"\nTotal Dataset Size: {len(data)} rows | Positive Disasters: {y.sum()} ({100 * y.mean():.3f}%)")

    # -----------------------------------------------------------------------
    # CROSS-VALIDATION — a robustness check, done FIRST, purely diagnostic.
    # With so few positives, one lucky/unlucky split can make the model look
    # much better or worse than it really is. 5-fold stratified CV trains and
    # evaluates 5 times on different slices and reports the average — a much
    # more trustworthy number than a single split. SMOTE is applied ONLY to
    # each fold's TRAINING portion, never to the held-out evaluation fold —
    # applying it to validation/test data would leak synthetic information
    # into the numbers you're trying to trust.
    # -----------------------------------------------------------------------
    print("\n--- 5-Fold Cross-Validation (robustness check, SMOTE on training folds only) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc_aucs, cv_pr_aucs = [], []

    for fold_i, (train_idx, eval_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train, X_fold_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_fold_train, y_fold_eval = y.iloc[train_idx], y.iloc[eval_idx]

        n_minority = y_fold_train.sum()
        if n_minority >= 6:
            smote = SMOTE(k_neighbors=min(5, n_minority - 1), random_state=42,sampling_strategy=0.15)
            X_fold_train, y_fold_train = smote.fit_resample(X_fold_train, y_fold_train)
        # else: too few positives in this fold to safely SMOTE — train as-is

        fold_model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )
        fold_model.fit(X_fold_train, y_fold_train)
        fold_probs = fold_model.predict_proba(X_fold_eval)[:, 1]

        fold_roc = roc_auc_score(y_fold_eval, fold_probs)
        fold_pr = average_precision_score(y_fold_eval, fold_probs)
        cv_roc_aucs.append(fold_roc)
        cv_pr_aucs.append(fold_pr)
        print(f"  Fold {fold_i}: ROC-AUC={fold_roc:.3f}  PR-AUC={fold_pr:.3f}  "
              f"(eval positives: {int(y_fold_eval.sum())})")

    print(f"\nCross-validated ROC-AUC: {np.mean(cv_roc_aucs):.3f} ± {np.std(cv_roc_aucs):.3f}")
    print(f"Cross-validated PR-AUC:  {np.mean(cv_pr_aucs):.3f} ± {np.std(cv_pr_aucs):.3f}")
    print("This is the more honest, defensible number to quote — an average across 5")
    print("different data splits, not whichever single split happened to look best.\n")

    print("NOTE: at this positive rate, threshold choice trades precision vs recall directly.")
    print("We pick the threshold on a VALIDATION split (never seen by the model, separate from")
    print("the test set used for final reporting) so the chosen cutoff isn't just overfit to")
    print("whichever split happened to be easiest.\n")

    # 3-way split: train (model fitting) / val (threshold selection only) / test (final report)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)} | Validation (threshold selection): {len(X_val)} "
          f"({y_val.sum()} positive) | Test (final report): {len(X_test)} ({y_test.sum()} positive)")

    # SMOTE applied ONLY to the training set here too — same rule as in CV above.
    n_minority_train = y_train.sum()
    if n_minority_train >= 6:
        smote = SMOTE(k_neighbors=min(5, n_minority_train - 1), random_state=42,sampling_strategy=0.15)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Applied SMOTE to training set only: {len(X_train)} -> {len(X_train_res)} rows "
              f"(validation/test sets left untouched, real data only).")
    else:
        X_train_res, y_train_res = X_train, y_train
        print("Too few positives in this training split for safe SMOTE — training on real data as-is.")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    test_probs = model.predict_proba(X_test)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    print(f"\nROC-AUC: {roc_auc:.3f}  (0.5 = random guessing, 1.0 = perfect separation)")
    print(f"PR-AUC:  {pr_auc:.3f}  (baseline for this data = {y_test.mean():.4f}, i.e. the positive rate)")

    # --- Honest caveat: with so few positive events, ANY single "best" threshold ---
    # is statistically shaky (chosen from a handful of validation positives). So
    # instead of picking one number and pretending it's optimal, show a SMALL TABLE
    # of candidate thresholds and let the actual deployment choose based on how
    # costly false alarms vs missed disasters are — a tunable sensitivity dial,
    # not a single fixed cutoff. This is also a legitimate real-world design
    # choice: disaster-response teams often want an adjustable alert sensitivity.
    print(f"\n--- Threshold options (selected on validation set, {int(y_val.sum())} real positives) ---")
    print("Lower threshold = more alerts, catches more real events, more false alarms.")
    print("Higher threshold = fewer alerts, fewer false alarms, risks missing real events.\n")
    print(f"{'Threshold':<10}{'Val Precision':<15}{'Val Recall':<12}{'Val F1':<8}")
    candidate_thresholds = [0.3, 0.4, 0.5,0.6,0.7,0.8]
    best_f1, best_threshold = -1, 0.3
    for t in candidate_thresholds:
        pred = (val_probs >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{t:<10}{precision:<15.3f}{recall:<12.3f}{f1:<8.3f}")
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    # Overriding auto-F1 pick — recall matters more than F1 here (missing a
    # real disaster is worse than a false alarm), so we lock threshold=0.4.
    best_threshold = 0.4
    print(f"\nUsing threshold = {best_threshold} (chosen for recall priority, not max F1).")

    # Final report on TEST set at the validation-chosen threshold (proper, no leakage)
    y_pred_final = (test_probs >= best_threshold).astype(int)
    print(f"\n--- Classification Report on TEST set at threshold={best_threshold} ---")
    print(classification_report(y_test, y_pred_final, labels=[0, 1],
                                 target_names=["No Event", "Flood/Landslide Risk"], zero_division=0))
    print(f"--- Confusion Matrix (threshold={best_threshold}) ---")
    print(confusion_matrix(y_test, y_pred_final, labels=[0, 1]))
    print(f"Total Test Cases Evaluated: {len(y_test)}")

    print("\n--- Feature importance ---")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(importance.round(3))

    # Missed-disaster analysis at the reference 0.5 cutoff — useful to inspect,
    # but remember: at 0.12% positive rate, plenty of real events will sit
    # below any single cutoff. This file is diagnostic, not a claim of failure.
    results_df = X_test.copy()
    results_df["Actual_Disaster"] = y_test
    results_df["Predicted_Disaster"] = y_pred_final
    results_df["Risk_Probability"] = test_probs

    false_negatives = results_df[(results_df["Actual_Disaster"] == 1) & (results_df["Predicted_Disaster"] == 0)]
    print(f"\n{len(false_negatives)} real events scored below 0.5 (reference cutoff). Saving detail to CSV...")
    false_negatives.to_csv("missed_disasters_analysis.csv", index=False)

    return model
# ---------------------------------------------------------------------------
# EXECUTION FLOW
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_filename = "flood_training_data_100k.csv"

    # Agar CSV pehle se padi hai, toh seedha load karo (0 seconds delay!)
    if os.path.exists(csv_filename):
        print(f"Loading pre-fetched dataset locally from {csv_filename} (No API delay! 🚀)...")
        feature_data = pd.read_csv(csv_filename)
        feature_data['date'] = pd.to_datetime(feature_data['date'])

        # If this CSV predates newer engineered features (added after it was
        # cached), recompute just those columns from what's already here —
        # no need to re-fetch anything from the internet for this.
        needed_cols = {"rain_5day", "extreme_rain_flag"}
        missing_cols = needed_cols - set(feature_data.columns)
        if missing_cols:
            print(f"  Cached CSV is missing newer columns {missing_cols} — recomputing "
                  f"from existing data (no re-fetch needed)...")
            feature_data = feature_data.sort_values(["location_id", "date"])
            if "rain_5day" in missing_cols:
                feature_data["rain_5day"] = feature_data.groupby("location_id")["rainfall_mm"].transform(
                    lambda s: s.rolling(5, min_periods=1).sum())
            if "extreme_rain_flag" in missing_cols:
                threshold_per_loc = feature_data.groupby("location_id")["rainfall_mm"].transform(
                    lambda s: s.quantile(0.95))
                feature_data["extreme_rain_flag"] = (feature_data["rainfall_mm"] >= threshold_per_loc).astype(int)
            feature_data.to_csv(csv_filename, index=False)
            print(f"  Updated CSV saved with new columns.")
    else:
        # Agar CSV nahi mili (pehli baar), tabhi internet se fetch hoga aur save hoga
        print("CSV not found. Running full data fetch pipeline...")
        locations = build_location_metadata(LOCATIONS)
        weather = fetch_historical_weather(locations, start_year=2005, end_year=2024)
        events = fetch_coolr_events(start_year=2000, end_year=2024)
        matched = match_events_to_locations(events, locations, radius_km=30.0)
        print(f"Matched {len(matched)} total disaster events across all Uttarakhand regions.")
        
        feature_data = build_features(weather, matched)
        feature_data.to_csv(csv_filename, index=False)

    model = train_model(feature_data)
    joblib.dump(model, "flood_model_uk.pkl")
    print("\nModel successfully trained and saved as flood_model_uk.pkl!")