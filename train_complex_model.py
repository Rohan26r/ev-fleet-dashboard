import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")
df = pd.read_csv('nev_fleet_dataset_v16_odo.csv')

print("Calculating odometer_km...")
df["odometer_km"] = df.groupby("vehicle_id")["trip_distance_km"].cumsum()

print("Encoding categorical variables...")
le_model = LabelEncoder()
df['model_encoded'] = le_model.fit_transform(df['model'])

df['location'] = df['location'].str.lower()
le_location = LabelEncoder()
df['location_encoded'] = le_location.fit_transform(df['location'])

# NEW COMPLEX FEATURES
features = [
    "battery_percentage",
    "battery_health_pct",
    "current_speed_kmph",
    "odometer_km",
    "model_encoded",
    "location_encoded",
    "battery_temp_c",        # Temperature drastically impacts lithium-ion performance
    "passenger_count",       # Extra weight impacts range
    "vehicle_weight_kg",     # Base mass of the vehicle
    "battery_capacity_kwh",  # Total energy capacity
    "motor_rpm"              # Efficiency curve of the motor at different RPMs
]

X = df[features]
y = df["estimated_range_km"]

print(f"Training on {len(features)} features...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- COMPLEX MODEL PERFORMANCE ---")
print(f"Mean Absolute Error: {mae:.2f} km")
print(f"R^2 Score: {r2:.4f}")

# Feature importance
importances = model.feature_importances_
importance_dict = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
print("\nFeature Importances:")
for f, i in importance_dict:
    print(f"  {f}: {i:.4f}")

with open('estimated_range_model_complex.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nComplex model saved to estimated_range_model_complex.pkl!")
