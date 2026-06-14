import os
import sqlite3
import pandas as pd
import numpy as np
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nev_fleet_dataset_v16_odo.csv')

# Global data frames & prediction model
df = pd.DataFrame()
model = None
emergency_alerts = []

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    
    # New driver_details table
    c.execute('''
        CREATE TABLE IF NOT EXISTS driver_details (
            driver_id TEXT PRIMARY KEY,
            driver_name TEXT NOT NULL,
            license_number TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            assigned_vehicle_id TEXT,
            base_salary_per_km REAL DEFAULT 8.0,
            created_date TEXT,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    ''')
    
    # New driver_trips table for trip assignments and salary calculation
    c.execute('''
        CREATE TABLE IF NOT EXISTS driver_trips (
            trip_assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id TEXT NOT NULL,
            vehicle_id TEXT,
            trip_date TEXT NOT NULL,
            destination_km REAL NOT NULL,
            salary_earned REAL NOT NULL,
            status TEXT DEFAULT 'Pending',
            created_at TEXT,
            FOREIGN KEY (driver_id) REFERENCES driver_details(driver_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_data_and_train_model():
    global df, model
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            
            # ── Standardize column names ────────────────────────
            column_mapping = {
                'battery_temp_c': 'battery_temperature_c',
                'battery_health_pct': 'battery_health',
                'current_speed_kmph': 'speed_kmph',
                'overspeeding_flag': 'overspeed_flag',
                'driving_efficiency_pct': 'driving_efficiency_km_kwh',
                'income_inr': 'income',
                'maintenance_cost_inr': 'maintenance_cost',
                'charge_cost_inr': 'charging_cost'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # ── Date parsing ────────────────────────
            if 'date_time' in df.columns:
                df['date'] = pd.to_datetime(df['date_time'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'])
            else:
                raise KeyError("Neither 'date', 'datetime', nor 'date_time' column found in dataset")
                
            # Extract month name from date
            df['month'] = df['date'].dt.strftime('%B')
            
            # ── Standardize status values (capitalize first letter) ────────────────
            if 'status' in df.columns:
                df['status'] = df['status'].str.capitalize()
                # Map Workshop to Garage (vehicles in workshop are in garage)
                df['status'] = df['status'].replace('Workshop', 'Garage')
            
            # ── Derive missing fields ────────────────
            
            # 1. battery_temperature_c
            if 'battery_temperature_c' not in df.columns:
                np.random.seed(42)
                base_temp = np.where(df['status'] == 'Running', 35.0, 25.0)
                speed_factor = df['speed_kmph'] * 0.1
                noise = np.random.normal(0, 2, size=len(df))
                df['battery_temperature_c'] = (base_temp + speed_factor + noise).clip(20.0, 55.0).round(2)
                
            # 2. body_type, total_seats, max_speed_kmph
            VEHICLE_SPECS = {
                ('Mahindra', 'BE 6'): ('SUV', 7, 200),
                ('Suzuki', 'e-Vitara'): ('SUV', 5, 160),
                ('Hyundai', 'Creta EV'): ('SUV', 5, 160),
                ('Tata', 'Nexon EV'): ('SUV', 5, 150),
                ('Tata', 'Tiago EV'): ('Hatchback', 5, 120),
                ('Tata', 'Xpres T EV'): ('Sedan', 4, 120)
            }
            
            if 'body_type' not in df.columns or 'total_seats' not in df.columns or 'max_speed_kmph' not in df.columns:
                def get_specs(row):
                    key = (row['brand'], row['model'])
                    return VEHICLE_SPECS.get(key, ('SUV', 5, 150))
                
                specs_applied = df.apply(get_specs, axis=1)
                
                if 'body_type' not in df.columns:
                    df['body_type'] = [s[0] for s in specs_applied]
                if 'total_seats' not in df.columns:
                    df['total_seats'] = [s[1] for s in specs_applied]
                if 'max_speed_kmph' not in df.columns:
                    df['max_speed_kmph'] = [s[2] for s in specs_applied]
                    
            # 3. soc_category
            if 'soc_category' not in df.columns:
                df['soc_category'] = pd.cut(
                    df['battery_percentage'],
                    bins=[-float('inf'), 20, 50, 80, float('inf')],
                    labels=['Critical', 'Low', 'Medium', 'High']
                ).astype(str)
            
            # 4. location_type (required for ML model)
            if 'location_type' not in df.columns:
                # Infer based on speed: Highway if speed > 80, City if 20-80, Stationary if < 20
                df['location_type'] = np.where(
                    df['speed_kmph'] > 80, 'Highway',
                    np.where(df['speed_kmph'] > 20, 'City', 'Stationary')
                )
            
            # Calculate charging cost if not present or all zero
            if 'charging_cost' not in df.columns or df['charging_cost'].sum() == 0:
                np.random.seed(42)
                charging_rate = np.random.uniform(16, 20, size=len(df))
                df['charging_cost'] = np.where(
                    df['charging_status'] == 1,
                    df['battery_capacity_kwh'] * charging_rate,
                    0.0
                )
            
            # Calculate theoretical max range at 100% battery health
            # Official manufacturer specifications for max range @ 100% health
            MANUFACTURER_MAX_RANGE = {
                ('Mahindra', 'BE 6'): 682,          # 79 kWh battery
                ('Suzuki', 'e-Vitara'): 550,        # Official spec
                ('Hyundai', 'Creta EV'): 473,       # 51.4 kWh battery
                ('Tata', 'Nexon EV'): 453,          # 40 kWh battery
                ('Tata', 'Tiago EV'): 315,          # 24 kWh battery
                ('Tata', 'Xpres T EV'): 395         # 26 kWh battery
            }
            
            def get_manufacturer_max_range(brand, model):
                key = (brand, model)
                return MANUFACTURER_MAX_RANGE.get(key, None)
            
            if 'max_range_100_health' not in df.columns:
                df['max_range_100_health'] = df.apply(
                    lambda row: get_manufacturer_max_range(row['brand'], row['model']),
                    axis=1
                )
            
            # For any vehicles not in the spec list, calculate from best recorded performance
            mask_missing = df['max_range_100_health'].isna()
            if mask_missing.any():
                if 'max_range_km' in df.columns:
                    # Fixed: Use transform to avoid FutureWarning
                    vehicle_max_ranges = df[mask_missing].groupby('vehicle_id', group_keys=False).agg({
                        'max_range_km': 'max'
                    }).reset_index()
                    vehicle_max_ranges.columns = ['vehicle_id', 'absolute_max_range']
                    
                    # Get health at max range for each vehicle
                    health_at_max = df[mask_missing].loc[df[mask_missing].groupby('vehicle_id')['max_range_km'].idxmax()]
                    vehicle_max_ranges['health_at_max'] = health_at_max['battery_health'].values
                    vehicle_max_ranges['calculated_max_100'] = vehicle_max_ranges['absolute_max_range'] * (100.0 / vehicle_max_ranges['health_at_max'])
                    
                    for _, vrow in vehicle_max_ranges.iterrows():
                        df.loc[df['vehicle_id'] == vrow['vehicle_id'], 'max_range_100_health'] = vrow['calculated_max_100']
                else:
                    df.loc[mask_missing, 'max_range_100_health'] = df.loc[mask_missing, 'battery_capacity_kwh'] * 8.0
            
            # Fill any remaining NaN values and cast to float
            df['max_range_100_health'] = df['max_range_100_health'].fillna(400.0).astype(float)

            # max_range_km = max_range_100_health scaled by current battery_health
            if 'max_range_km' not in df.columns:
                df['max_range_km'] = (df['max_range_100_health'] * df['battery_health'] / 100.0).round(2)

            # driving_efficiency_km_kwh (km per kWh) – avoid division by zero
            if 'driving_efficiency_km_kwh' not in df.columns:
                df['driving_efficiency_km_kwh'] = np.where(
                    df['energy_consumed_kwh'] > 0,
                    (df['trip_distance_km'] / df['energy_consumed_kwh']).round(3),
                    0.0
                )

            # regenerative_braking_kwh – approximate as 10 % of energy consumed while running
            if 'regenerative_braking_kwh' not in df.columns:
                df['regenerative_braking_kwh'] = np.where(
                    df['status'] == 'Running',
                    (df['energy_consumed_kwh'] * 0.10).round(3),
                    0.0
                )

            # is_peak_hour – 1 if hour is between 7-10 or 17-20
            if 'is_peak_hour' not in df.columns:
                hour = df['date'].dt.hour
                df['is_peak_hour'] = ((hour.between(7, 10)) | (hour.between(17, 20))).astype(int)

            # is_weekend
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

            # trip_count – count records where a trip occurred (distance > 0)
            if 'trip_count' not in df.columns:
                df['trip_count'] = (df['trip_distance_km'] > 0).astype(int)

            print(f"Dataset loaded. Shape: {df.shape}")
            print(f"Charging events: {(df['charging_status'] == 1).sum()}")
            print(f"Columns available: {list(df.columns)}")

            # ── Load pre-trained ML model ────────────────
            import pickle
            
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'estimated_range_model_complex.pkl')
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✓ Pre-trained model loaded from: {model_path}")
                except Exception as ex:
                    print(f"✗ Failed to load model from pkl: {ex}")
                    model = None
            else:
                print(f"✗ Model file not found at: {model_path}")
                model = None
        else:
            print(f"Data file not found at: {DATA_PATH}")
    except Exception as e:
        import traceback
        print(f"Error loading dataset or model: {e}")
        traceback.print_exc()

# Helper to map email -> Driver ID (e.g. driver2@rx7.com -> DRV002)
def get_driver_id_from_email(email):
    digits = re.findall(r'\d+', email)
    if digits:
        num = int(digits[0])
        return f"DRV{num:03d}"
    return "DRV001"

# ====================================================================
# HELPER FUNCTIONS FOR DRIVER NAMES AND VEHICLE MODELS
# ====================================================================

def get_driver_name(driver_id):
    """Get driver name from driver_id"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT driver_name FROM driver_details WHERE driver_id = ?", (driver_id,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else driver_id
    except:
        return driver_id

def get_vehicle_model(vehicle_id):
    """Get vehicle brand + model from vehicle_id using dataset"""
    try:
        vehicle_data = df[df['vehicle_id'] == vehicle_id]
        if not vehicle_data.empty:
            brand = vehicle_data.iloc[0]['brand']
            model = vehicle_data.iloc[0]['model']
            return f"{brand} {model}"
        return vehicle_id
    except:
        return vehicle_id

def enrich_data_with_names(data_list, id_field='driver_id', name_field='driver_name', vehicle_field=None):
    """
    Enrich list of dictionaries with driver names and vehicle models
    Args:
        data_list: list of dicts
        id_field: field containing driver_id
        name_field: new field name for driver name
        vehicle_field: field containing vehicle_id (optional)
    """
    for item in data_list:
        if id_field in item and item[id_field]:
            item[name_field] = get_driver_name(item[id_field])
        
        if vehicle_field and vehicle_field in item and item[vehicle_field]:
            item['vehicle_model'] = get_vehicle_model(item[vehicle_field])
    
    return data_list

# Call startup functions
init_db()
load_data_and_train_model()

# --- Auth Routes ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/auth', methods=['POST'])
def auth():
    data = request.json
    action = data.get('action')
    email = data.get('email')
    password = data.get('pass')
    role = data.get('role')

    if not email or not password or not action:
        return jsonify({"error": "Missing required fields"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        if action == 'signup':
            c.execute('INSERT INTO users (email, password, role) VALUES (?, ?, ?)', (email, password, role))
            conn.commit()
            return jsonify({"message": f"Successfully registered as {role}!"}), 201

        elif action == 'login':
            c.execute('SELECT * FROM users WHERE email=? AND password=? AND role=?', (email, password, role))
            user = c.fetchone()
            if user:
                return jsonify({"message": f"Welcome back, {email}!"}), 200
            else:
                return jsonify({"error": "Invalid credentials or incorrect role"}), 401
        else:
            return jsonify({"error": "Invalid action"}), 400
    except sqlite3.IntegrityError:
        return jsonify({"error": "User with this email already exists"}), 409
    finally:
        conn.close()

@app.route('/api/users', methods=['GET'])
def get_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('SELECT email, role FROM users')
        users = c.fetchall()
        user_list = [{"email": u[0], "role": u[1]} for u in users]
        return jsonify({"users": user_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# ====================================================================
# ADMIN ENDPOINTS
# ====================================================================

@app.route('/api/admin/overview', methods=['GET'])
def admin_overview():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    # Latest record per vehicle
    latest = df.sort_values('date').groupby('vehicle_id').last().reset_index()
    
    total_vehicles = len(latest)
    vehicles_running = int((latest['status'] == 'Running').sum())
    vehicles_charging = int((latest['status'] == 'Charging').sum())
    vehicles_garage = int((latest['status'] == 'Garage').sum())
    
    avg_battery = float(latest['battery_percentage'].mean())
    avg_health = float(latest['battery_health'].mean())
    avg_temp = float(latest['battery_temperature_c'].mean())
    
    # Financial metrics over the entire dataset
    total_revenue = float(df['income'].sum())
    total_expenses = float((df['charging_cost'] + df['maintenance_cost']).sum())
    total_profit = total_revenue - total_expenses
    
    # EV-specific metrics
    total_distance = float(df['trip_distance_km'].sum())
    total_energy = float(df['energy_consumed_kwh'].sum())
    total_overspeeds = int(df['overspeed_flag'].sum())
    avg_efficiency = float(df[df['driving_efficiency_km_kwh'] > 0]['driving_efficiency_km_kwh'].mean())
    
    return jsonify({
        "total_vehicles": total_vehicles,
        "vehicles_running": vehicles_running,
        "vehicles_charging": vehicles_charging,
        "vehicles_garage": vehicles_garage,
        "avg_battery": round(avg_battery, 2),
        "avg_battery_health": round(avg_health, 2),
        "avg_battery_temp": round(avg_temp, 2),
        "total_revenue": round(total_revenue, 2),
        "total_expenses": round(total_expenses, 2),
        "total_profit": round(total_profit, 2),
        "total_distance_km": round(total_distance, 2),
        "total_energy_kwh": round(total_energy, 2),
        "total_overspeeds": total_overspeeds,
        "avg_efficiency_km_kwh": round(avg_efficiency, 2)
    }), 200

@app.route('/api/admin/ev-metrics', methods=['GET'])
def ev_metrics():
    """Get EV-specific metrics like body types, SOC distribution, location analytics"""
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    latest = df.sort_values('date').groupby('vehicle_id').last().reset_index()
    
    # Body type distribution
    body_types = latest['body_type'].value_counts().to_dict()
    
    # SOC category distribution
    soc_categories = latest['soc_category'].value_counts().to_dict()
    
    # Location type analytics
    location_stats = df.groupby('location_type').agg({
        'trip_distance_km': 'sum',
        'energy_consumed_kwh': 'sum',
        'overspeed_flag': 'sum'
    }).to_dict('index')
    
    # Peak hour vs off-peak analytics
    peak_trips = int(df[df['is_peak_hour'] == 1]['trip_count'].sum())
    offpeak_trips = int(df[df['is_peak_hour'] == 0]['trip_count'].sum())
    
    # Weekend vs weekday
    weekend_distance = float(df[df['is_weekend'] == 1]['trip_distance_km'].sum())
    weekday_distance = float(df[df['is_weekend'] == 0]['trip_distance_km'].sum())
    
    return jsonify({
        "body_types": body_types,
        "soc_categories": soc_categories,
        "location_stats": location_stats,
        "peak_trips": peak_trips,
        "offpeak_trips": offpeak_trips,
        "weekend_distance_km": round(weekend_distance, 2),
        "weekday_distance_km": round(weekday_distance, 2)
    }), 200

@app.route('/api/admin/vehicles', methods=['GET'])
def admin_vehicles():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    # Latest record per vehicle
    latest = df.sort_values('date').groupby('vehicle_id').last().reset_index()
    latest = latest.sort_values('vehicle_id')
    
    vehicles_list = []
    for _, row in latest.iterrows():
        vehicles_list.append({
            "vehicle_id": row['vehicle_id'],
            "brand": row['brand'],
            "model": row['model'],
            "body_type": row['body_type'],
            "total_seats": int(row['total_seats']),
            "driver_id": row['driver_id'],
            "battery_percentage": round(float(row['battery_percentage']), 2),
            "battery_health": round(float(row['battery_health']), 2),
            "battery_temperature_c": round(float(row['battery_temperature_c']), 2),
            "soc_category": row['soc_category'],
            "estimated_range_km": round(float(row['estimated_range_km']), 2),
            "max_range_km": round(float(row['max_range_km']), 2),
            "max_range_100_health": round(float(row['max_range_100_health']), 2),
            "status": row['status']
        })
    return jsonify(vehicles_list), 200

@app.route('/api/admin/vehicle/<vehicle_id>', methods=['GET'])
def admin_vehicle_details(vehicle_id):
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    v_df = df[df['vehicle_id'] == vehicle_id].copy()
    if v_df.empty:
        return jsonify({"error": f"Vehicle {vehicle_id} not found"}), 404
        
    latest_v = v_df.sort_values('date').iloc[-1]
    
    # Monthly analytics
    monthly_stats = v_df.groupby('month').agg({
        'income': 'sum',
        'charging_cost': 'sum',
        'maintenance_cost': 'sum',
        'overspeed_flag': 'sum'
    }).to_dict('index')
    
    # Count charging status = 'Charging' per month
    monthly_charging = v_df[v_df['status'] == 'Charging'].groupby('month').size().to_dict()
    
    months = ['March', 'April', 'May', 'June']
    monthly_analytics = []
    for m in months:
        m_data = monthly_stats.get(m, {'income': 0.0, 'charging_cost': 0.0, 'maintenance_cost': 0.0, 'overspeed_flag': 0})
        rev = float(m_data['income'])
        exp = float(m_data['charging_cost'] + m_data['maintenance_cost'])
        charging_cnt = int(monthly_charging.get(m, 0))
        overspeed_cnt = int(m_data['overspeed_flag'])
        
        monthly_analytics.append({
            "month": m,
            "revenue": round(rev, 2),
            "expense": round(exp, 2),
            "charging_count": charging_cnt,
            "overspeed_count": overspeed_cnt
        })
        
    # Daily analytics
    v_df['date_str'] = v_df['date'].dt.strftime('%Y-%m-%d')
    daily_stats = v_df.groupby('date_str').agg({
        'income': 'sum',
        'charging_cost': 'sum',
        'maintenance_cost': 'sum',
        'overspeed_flag': 'sum'
    }).to_dict('index')
    
    daily_charging = v_df[v_df['status'] == 'Charging'].groupby('date_str').size().to_dict()
    
    daily_analytics = []
    for d_str in sorted(list(daily_stats.keys())):
        d_data = daily_stats[d_str]
        rev = float(d_data['income'])
        exp = float(d_data['charging_cost'] + d_data['maintenance_cost'])
        prof = rev - exp
        charging_cnt = int(daily_charging.get(d_str, 0))
        overspeed_cnt = int(d_data['overspeed_flag'])
        
        daily_analytics.append({
            "date": d_str,
            "revenue": round(rev, 2),
            "expense": round(exp, 2),
            "profit": round(prof, 2),
            "charging_count": charging_cnt,
            "overspeed_count": overspeed_cnt
        })
        
    total_distance = float(v_df['trip_distance_km'].sum())
    total_passengers = int(v_df['passenger_count'].sum())
    total_energy = float(v_df['energy_consumed_kwh'].sum())
    total_regen = float(v_df['regenerative_braking_kwh'].sum())
    avg_efficiency = float(v_df[v_df['driving_efficiency_km_kwh'] > 0]['driving_efficiency_km_kwh'].mean())
    
    # NEW: Calculate charging statistics
    # Get ride_complete from the latest record (current date)
    total_trips = int(latest_v['ride_complete'])
    vehicle_charged_count = int((v_df['status'] == 'Charging').sum())
    
    # Calculate total and average charging cost
    charging_records = v_df[v_df['charging_cost'] > 0]
    total_charging_cost = float(charging_records['charging_cost'].sum())
    avg_charging_cost = float(charging_records['charging_cost'].mean()) if not charging_records.empty else 0.0
    
    return jsonify({
        "vehicle_id": vehicle_id,
        "brand": latest_v['brand'],
        "model": latest_v['model'],
        "body_type": latest_v['body_type'],
        "total_seats": int(latest_v['total_seats']),
        "driver_id": latest_v['driver_id'],
        "battery_capacity_kwh": float(latest_v['battery_capacity_kwh']),
        "vehicle_weight_kg": float(latest_v['vehicle_weight_kg']),
        "max_speed_kmph": float(latest_v['max_speed_kmph']),
        "battery_percentage": round(float(latest_v['battery_percentage']), 2),
        "battery_health": round(float(latest_v['battery_health']), 2),
        "battery_temperature_c": round(float(latest_v['battery_temperature_c']), 2),
        "soc_category": latest_v['soc_category'],
        "estimated_range_km": round(float(latest_v['estimated_range_km']), 2),
        "max_range_km": round(float(latest_v['max_range_km']), 2),
        "max_range_100_health": round(float(latest_v['max_range_100_health']), 2),
        "status": latest_v['status'],
        
        "monthly_analytics": monthly_analytics,
        "daily_analytics": daily_analytics,
        
        "total_distance": round(total_distance, 2),
        "total_passengers": total_passengers,
        "total_trips": total_trips,
        "vehicle_charged_count": vehicle_charged_count,
        "total_charging_cost": round(total_charging_cost, 2),
        "avg_charging_cost": round(avg_charging_cost, 2),
        "total_energy_consumed_kwh": round(total_energy, 2),
        "total_regen_braking_kwh": round(total_regen, 2),
        "avg_efficiency_km_kwh": round(avg_efficiency, 2),
        "current_range": round(float(latest_v['estimated_range_km']), 2),
        "current_battery": round(float(latest_v['battery_percentage']), 2),
        "current_health": round(float(latest_v['battery_health']), 2)
    }), 200

@app.route('/api/admin/brands', methods=['GET'])
def admin_brands():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    brands = sorted(list(df['brand'].unique()))
    return jsonify(brands), 200

@app.route('/api/admin/brand/<brand_name>', methods=['GET'])
def admin_brand_details(brand_name):
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    b_df = df[df['brand'].str.lower() == brand_name.lower()]
    if b_df.empty:
        return jsonify({"error": f"Brand {brand_name} not found"}), 404
        
    # Monthly Analytics
    monthly_stats = b_df.groupby('month').agg({
        'income': 'sum',
        'charging_cost': 'sum',
        'maintenance_cost': 'sum'
    }).to_dict('index')
    
    months = ['March', 'April', 'May', 'June']
    monthly_analytics = []
    for m in months:
        m_data = monthly_stats.get(m, {'income': 0.0, 'charging_cost': 0.0, 'maintenance_cost': 0.0})
        rev = float(m_data['income'])
        exp = float(m_data['charging_cost'] + m_data['maintenance_cost'])
        monthly_analytics.append({
            "month": m,
            "revenue": round(rev, 2),
            "expense": round(exp, 2)
        })
        
    total_revenue = float(b_df['income'].sum())
    total_expenses = float((b_df['charging_cost'] + b_df['maintenance_cost']).sum())
    total_profit = total_revenue - total_expenses
    
    latest_b = b_df.sort_values('date').groupby('vehicle_id').last()
    avg_health = float(latest_b['battery_health'].mean()) if not latest_b.empty else 0.0
    avg_range = float(latest_b['estimated_range_km'].mean()) if not latest_b.empty else 0.0
    
    charging_events = int((b_df['status'] == 'Charging').sum())
    overspeed_events = int(b_df['overspeed_flag'].sum())
    
    return jsonify({
        "brand": brand_name,
        "monthly_analytics": monthly_analytics,
        "total_revenue": round(total_revenue, 2),
        "total_expenses": round(total_expenses, 2),
        "total_profit": round(total_profit, 2),
        "avg_battery_health": round(avg_health, 2),
        "avg_range": round(avg_range, 2),
        "charging_events": charging_events,
        "overspeed_events": overspeed_events
    }), 200

@app.route('/api/admin/drivers', methods=['GET'])
def admin_drivers():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    latest_driver_records = df.sort_values('date').groupby('driver_id').last().reset_index()
    latest_driver_records = latest_driver_records.sort_values('driver_id')
    
    drivers_list = []
    for _, row in latest_driver_records.iterrows():
        drivers_list.append({
            "driver_id": row['driver_id'],
            "assigned_vehicle": row['vehicle_id']
        })
    return jsonify(drivers_list), 200

@app.route('/api/admin/driver/<driver_id>', methods=['GET'])
def admin_driver_details(driver_id):
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    d_df = df[df['driver_id'] == driver_id]
    if d_df.empty:
        return jsonify({"error": f"Driver {driver_id} not found"}), 404
        
    latest_d = d_df.sort_values('date').iloc[-1]
    assigned_vehicle = latest_d['vehicle_id']
    vehicle_model = latest_d['model']
    
    total_trips = int((d_df['trip_distance_km'] > 0).sum())
    total_distance = float(d_df['trip_distance_km'].sum())
    overspeed_count = int(d_df['overspeed_flag'].sum())
    
    driving_records = d_df[d_df['speed_kmph'] > 0]
    avg_speed = float(driving_records['speed_kmph'].mean()) if not driving_records.empty else 0.0
    
    revenue_generated = float(d_df['income'].sum())
    charging_events = int((d_df['status'] == 'Charging').sum())
    
    eco_score = max(0, min(100, 100 - (overspeed_count * 2)))
    
    # Extract monthly overspeed count for past 3 months (April, May, June)
    monthly_overspeeds = d_df.groupby('month')['overspeed_flag'].sum().to_dict()
    past_3_months = ['April', 'May', 'June']
    overspeed_trend = []
    for m in past_3_months:
        overspeed_trend.append({
            "month": m,
            "overspeed_count": int(monthly_overspeeds.get(m, 0))
        })
    
    return jsonify({
        "driver_id": driver_id,
        "assigned_vehicle": assigned_vehicle,
        "vehicle_model": vehicle_model,
        "total_trips": total_trips,
        "total_distance": round(total_distance, 2),
        "overspeed_count": overspeed_count,
        "average_speed": round(avg_speed, 2),
        "revenue_generated": round(revenue_generated, 2),
        "charging_events": charging_events,
        "eco_score": eco_score,
        "overspeed_trend": overspeed_trend
    }), 200

@app.route('/api/admin/charging', methods=['GET'])
def admin_charging():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    total_events = int((df['status'] == 'Charging').sum())
    total_cost = float(df['charging_cost'].sum())
    
    charging_rows = df[df['charging_cost'] > 0]
    avg_cost = float(charging_rows['charging_cost'].mean()) if not charging_rows.empty else 0.0
    
    # Most Charged Vehicle (by count of charging records)
    most_charged_id = "N/A"
    charging_counts = df[df['status'] == 'Charging'].groupby('vehicle_id').size()
    if not charging_counts.empty:
        most_charged_id = charging_counts.idxmax()
        
    # Top 5 vehicles by charging cost
    top_vehicles = df.groupby('vehicle_id')['charging_cost'].sum().reset_index()
    top_5 = top_vehicles.sort_values('charging_cost', ascending=False).head(5)
    top_5_list = []
    for _, row in top_5.iterrows():
        top_5_list.append({
            "vehicle_id": row['vehicle_id'],
            "charging_cost": round(float(row['charging_cost']), 2)
        })
        
    # Monthly trend
    monthly_costs = df.groupby('month')['charging_cost'].sum().to_dict()
    months = ['March', 'April', 'May', 'June']
    trend = []
    for m in months:
        trend.append({
            "month": m,
            "cost": round(float(monthly_costs.get(m, 0.0)), 2)
        })
        
    return jsonify({
        "total_charging_events": total_events,
        "total_charging_cost": round(total_cost, 2),
        "average_charging_cost": round(avg_cost, 2),
        "most_charged_vehicle": most_charged_id,
        "top_vehicles": top_5_list,
        "monthly_trend": trend
    }), 200

@app.route('/api/admin/alerts', methods=['GET'])
def admin_alerts():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    # Use latest record of each vehicle
    latest = df.sort_values('date').groupby('vehicle_id').last().reset_index()
    
    critical_battery = []
    degraded_health = []
    in_garage = []
    overspeeding = []
    
    for _, row in latest.iterrows():
        v_id = row['vehicle_id']
        brand_model = f"{row['brand']} {row['model']}"
        driver = row['driver_id']
        
        if row['battery_percentage'] < 20:
            critical_battery.append({
                "vehicle_id": v_id,
                "value": round(float(row['battery_percentage']), 1),
                "message": f"Critical Battery: {v_id} ({brand_model}) is at {row['battery_percentage']:.1f}%"
            })
            
        if row['battery_health'] < 70:
            degraded_health.append({
                "vehicle_id": v_id,
                "value": round(float(row['battery_health']), 1),
                "message": f"Degraded Health: {v_id} battery health is at {row['battery_health']:.1f}%"
            })
            
        if row['status'] == 'Garage':
            in_garage.append({
                "vehicle_id": v_id,
                "message": f"In Garage: {v_id} is checked in for repairs/servicing."
            })
            
        if row['overspeed_flag'] == 1:
            overspeeding.append({
                "vehicle_id": v_id,
                "speed": round(float(row['speed_kmph']), 1),
                "message": f"Overspeed warning: {v_id} driven by {driver} detected at {row['speed_kmph']:.1f} km/h"
            })
            
    return jsonify({
        "critical_battery": critical_battery,
        "degraded_health": degraded_health,
        "in_garage": in_garage,
        "overspeeding": overspeeding,
        "emergencies": emergency_alerts
    }), 200


# ====================================================================
# DRIVER ENDPOINTS
# ====================================================================

@app.route('/api/driver/dashboard/<driver_id>', methods=['GET'])
def driver_dashboard_details(driver_id):
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    # Get driver details from driver_details table
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT driver_name, license_number, email, base_salary_per_km 
        FROM driver_details 
        WHERE email = ? OR driver_id = ?
    ''', (driver_id, driver_id))
    driver_row = c.fetchone()
    
    driver_name = driver_row[0] if driver_row else "Driver"
    license_number = driver_row[1] if driver_row else "N/A"
    driver_email = driver_row[2] if driver_row else driver_id
    base_salary_rate = driver_row[3] if driver_row else 2.30
    
    # Get total salary earned from trip_salaries table
    c.execute('''
        SELECT COALESCE(SUM(salary_amount), 0.0)
        FROM trip_salaries
        WHERE driver_id = ?
    ''', (driver_id,))
    total_salary_earned = c.fetchone()[0]
    conn.close()
        
    d_df = df[df['driver_id'] == driver_id]
    if d_df.empty:
        # If driver has no records in dataset, try fallback mapping from email
        resolved_id = get_driver_id_from_email(driver_id)
        d_df = df[df['driver_id'] == resolved_id]
        if d_df.empty:
            return jsonify({"error": f"Driver {driver_id} not found"}), 404
        driver_id = resolved_id
            
    latest_rec = d_df.sort_values('date').iloc[-1]
    v_id = latest_rec['vehicle_id']
    
    # Total distance traveled (from dataset)
    total_distance_traveled = float(d_df['trip_distance_km'].sum())
    
    # 1. My Vehicle Details
    vehicle_info = {
        "vehicle_id": v_id,
        "brand": latest_rec['brand'],
        "model": latest_rec['model'],
        "battery_capacity_kwh": float(latest_rec['battery_capacity_kwh']),
        "current_battery": round(float(latest_rec['battery_percentage']), 2),
        "battery_health": round(float(latest_rec['battery_health']), 2),
        "current_range": round(float(latest_rec['estimated_range_km']), 2),
        "current_status": latest_rec['status'],
        "odometer_km": round(total_distance_traveled, 2),
        "drive_type": latest_rec['location_type'],
        "charge_count": int(latest_rec['charge_count'])
    }
    
    # 2. Trip Summary (Today's totals = totals on the last active date)
    latest_date = latest_rec['date'].date()
    today_df = d_df[d_df['date'].dt.date == latest_date]
    
    todays_distance = float(today_df['trip_distance_km'].sum())
    todays_trips = int((today_df['trip_distance_km'] > 0).sum())
    passengers_transported = int(today_df['passenger_count'].sum())
    
    # Calculate predominant drive type for today
    if not today_df.empty:
        location_counts = today_df['location_type'].value_counts()
        todays_drive_type = location_counts.index[0] if len(location_counts) > 0 else "Mixed"
    else:
        todays_drive_type = "--"
    
    trip_summary = {
        "todays_distance_km": round(todays_distance, 2),
        "todays_trips": todays_trips,
        "passengers_transported": passengers_transported,
        "revenue_generated_today": round(float(today_df['income'].sum()), 2),
        "todays_drive_type": todays_drive_type
    }
    
    # 3. Charging Information
    total_charging_count = int((d_df['status'] == 'Charging').sum())
    
    # Find last charging event
    charging_events = d_df[(d_df['status'] == 'Charging') | (d_df['charging_cost'] > 0)].sort_values('date')
    last_charging_cost = 0.0
    if not charging_events.empty:
        last_charging_cost = float(charging_events.iloc[-1]['charging_cost'])
        
    charging_info = {
        "charging_status": int(latest_rec['charging_status']),
        "last_charging_cost": round(last_charging_cost, 2),
        "total_charging_count": total_charging_count
    }
    
    # 4. Driving Behaviour
    overspeed_count = int(d_df['overspeed_flag'].sum())
    driving_recs = d_df[d_df['speed_kmph'] > 0]
    avg_speed = float(driving_recs['speed_kmph'].mean()) if not driving_recs.empty else 0.0
    eco_score = max(0, min(100, 100 - (overspeed_count * 2)))
    
    driving_behaviour = {
        "overspeed_count": overspeed_count,
        "average_speed_kmph": round(avg_speed, 2),
        "maximum_speed_kmph": float(driving_recs['speed_kmph'].max()) if not driving_recs.empty else 0.0,
        "eco_score": eco_score
    }
    
    # 5. Maintenance & Recommendations (Section 6)
    alerts = []
    if latest_rec['battery_health'] < 70:
        alerts.append({
            "type": "danger",
            "message": "Critical Battery Degradation: Battery Health is below 70%. Schedule pack replacement."
        })
    elif latest_rec['battery_health'] < 80:
        alerts.append({
            "type": "warning",
            "message": "Battery degradation alert: Health is below 80%. Battery cell balancing recommended."
        })
        
    if latest_rec['status'] == 'Garage':
        alerts.append({
            "type": "info",
            "message": "Garage Alert: Vehicle is currently checked in for servicing."
        })
        
    if latest_rec['battery_percentage'] < 20:
        alerts.append({
            "type": "danger",
            "message": "Low Battery Warning: Battery is below 20%. Please head to the nearest charging dock."
        })
        
    # Service recommendation based on distance and overspeed events
    total_dist = d_df['trip_distance_km'].sum()
    if total_dist > 1500:
        alerts.append({
            "type": "info",
            "message": "Service Recommendation: Fleet distance has exceeded 1,500 km. Schedule routine tire rotation."
        })
        
    if not alerts:
        alerts.append({
            "type": "success",
            "message": "All vehicle systems nominal. Continue driving safely!"
        })
        
    return jsonify({
        "driver_id": driver_id,
        "driver_name": driver_name,
        "license_number": license_number,
        "total_salary_earned": round(total_salary_earned, 2),
        "vehicle_info": vehicle_info,
        "trip_summary": trip_summary,
        "charging_info": charging_info,
        "driving_behaviour": driving_behaviour,
        "maintenance_alerts": alerts
    }), 200

@app.route('/api/driver/emergency', methods=['POST'])
def trigger_emergency():
    try:
        data = request.json
        driver_id = data.get('driver_id')
        if not driver_id:
            return jsonify({"error": "Missing driver_id"}), 400
            
        # Resolve driver id if email is passed
        if '@' in driver_id:
            driver_id_resolved = get_driver_id_from_email(driver_id)
        else:
            driver_id_resolved = driver_id
            
        # Get vehicle ID for the driver
        d_df = df[df['driver_id'] == driver_id_resolved]
        if not d_df.empty:
            vehicle_id = d_df.sort_values('date').iloc[-1]['vehicle_id']
        else:
            vehicle_id = "Unknown"
            
        import datetime
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        emergency_alert = {
            "driver_id": driver_id_resolved,
            "vehicle_id": vehicle_id,
            "timestamp": now_str,
            "message": f"EMERGENCY: Driver {driver_id_resolved} (Vehicle {vehicle_id}) has triggered an emergency alert!"
        }
        
        emergency_alerts.append(emergency_alert)
        print(f"Emergency Alert added: {emergency_alert}")
        
        return jsonify({"message": "Emergency alert submitted successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/admin/clear-emergencies', methods=['POST'])
def clear_emergencies():
    global emergency_alerts
    emergency_alerts.clear()
    return jsonify({"message": "All emergencies resolved and cleared."}), 200

@app.route('/api/predict-range', methods=['POST'])
@app.route('/api/admin/range-predict', methods=['POST'])
def admin_range_predict():
    if model is None:
        return jsonify({"error": "Prediction model not trained"}), 500
    try:
        import pandas as pd
        data = request.json
        batt_pct = float(data.get('battery_percentage', 100))
        health = float(data.get('battery_health', 100))
        speed = float(data.get('speed_kmph', 0))
        odometer = float(data.get('odometer_km', 10000.0))
        loc_type = data.get('location_type', 'City')
        vehicle_model = data.get('vehicle_model', 'Tiago EV')
        
        location_map = {'City': 0, 'Garage': 1, 'Highway': 2}
        location_encoded = location_map.get(loc_type, 0)
        
        model_name_map = {'BE 6': 0, 'Creta': 1, 'Tiago EV': 2, 'Vitara': 3, 'Xpres T EV': 4}
        model_encoded = model_name_map.get(vehicle_model, 2)
        
        temp = float(data.get('battery_temperature_c', 35.0))
        passengers = int(data.get('passenger_count', 1))
        weight = float(data.get('vehicle_weight_kg', 1200.0))
        capacity = float(data.get('battery_capacity_kwh', 40.0))
        rpm = float(data.get('motor_rpm', 0.0))

        features = [
            "battery_percentage", "battery_health_pct", "current_speed_kmph", 
            "odometer_km", "model_encoded", "location_encoded",
            "battery_temp_c", "passenger_count", "vehicle_weight_kg", 
            "battery_capacity_kwh", "motor_rpm"
        ]
        df_input = pd.DataFrame([[
            batt_pct, health, speed, odometer, model_encoded, location_encoded,
            temp, passengers, weight, capacity, rpm
        ]], columns=features)
        
        predicted_range = float(model.predict(df_input)[0])
        
        return jsonify({
            "predicted_range_km": round(predicted_range, 2)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/driver/range-predict', methods=['POST'])
def driver_range_predict():
    """
    Predict estimated range using pre-trained model (7 features)
    Training format from est_range.ipynb:
    - Features: battery_percentage, battery_health_pct, current_speed_kmph, 
                odometer_km, max_range_km, model, location
    - Model: LinearRegression
    - Encoding: LabelEncoder (alphabetical)
    
    Frontend inputs: battery_percentage, battery_health, speed_kmph, odometer_km, location_type, vehicle_model
    Backend fetches: max_range_km from dataset
    """
    if model is None:
        return jsonify({"error": "Prediction model not loaded"}), 500
        
    try:
        import pandas as pd
        
        data = request.json
        
        # Frontend inputs
        battery_percentage = float(data.get('battery_percentage'))
        battery_health = float(data.get('battery_health'))
        speed_kmph = float(data.get('speed_kmph'))
        odometer_km = float(data.get('odometer_km'))
        charge_count = int(data.get('charge_count', 0))  # NEW: charge count
        location_type = data.get('location_type', 'City')
        vehicle_model = data.get('vehicle_model', 'Tata Tiago EV')
        
        # INPUT VALIDATION - Reject invalid values
        if battery_percentage < 0 or battery_percentage > 100:
            return jsonify({
                "error": "Battery percentage must be between 0 and 100",
                "predicted_range_km": 0.0
            }), 400
        
        if battery_health < 0 or battery_health > 100:
            return jsonify({
                "error": "Battery health must be between 0 and 100",
                "predicted_range_km": 0.0
            }), 400
        
        if speed_kmph < 0 or speed_kmph > 300:
            return jsonify({
                "error": "Speed must be between 0 and 300 km/h",
                "predicted_range_km": 0.0
            }), 400
        
        if odometer_km < 0:
            return jsonify({
                "error": "Odometer cannot be negative",
                "predicted_range_km": 0.0
            }), 400
        
        if charge_count < 0:
            return jsonify({
                "error": "Charge count cannot be negative",
                "predicted_range_km": 0.0
            }), 400
        
        # CRITICAL CONSTRAINTS - Return 0 for impossible conditions
        if battery_percentage <= 0:
            return jsonify({
                "predicted_range_km": 0.0,
                "message": "No battery charge remaining",
                "vehicle_specs_used": {"model": vehicle_model, "max_range_km": 0}
            }), 200
        
        if battery_health <= 10:
            return jsonify({
                "predicted_range_km": 0.0,
                "message": "Battery health critically low - needs replacement",
                "vehicle_specs_used": {"model": vehicle_model, "max_range_km": 0}
            }), 200
        
        # Map frontend model names to dataset model names
        model_map = {
            'mahindra_be6': 'BE 6',
            'tata_nexon': 'Tiago EV',
            'tata_tiago': 'Tiago EV',
            'tata_xpres': 'Xpres T EV',
            'hyundai_creta': 'Creta',
            'suzuki_vitara': 'Vitara'
        }
        
        dataset_model_name = model_map.get(vehicle_model, 'Tiago EV')
        
        # Static max range per model (manufacturer specifications at 100% health)
        model_max_range_static = {
            'BE 6': 682.0,
            'Creta': 473.0,
            'Tiago EV': 315.0,
            'Vitara': 550.0,
            'Xpres T EV': 395.0
        }
        
        # Get static max range for this model
        max_range_km = model_max_range_static.get(dataset_model_name, 315.0)
        
        # Encode model (MUST MATCH training - alphabetical LabelEncoder)
        # BE 6=0, Creta=1, Tiago EV=2, Vitara=3, Xpres T EV=4
        model_name_map = {
            'BE 6': 0, 'Creta': 1, 'Tiago EV': 2, 
            'Vitara': 3, 'Xpres T EV': 4
        }
        model_encoded = model_name_map.get(dataset_model_name, 2)
        
        # Encode location (MUST MATCH training - alphabetical lowercase)
        # city=0, garage=1, highway=2
        location_map = {'City': 0, 'Highway': 2}
        location_encoded = location_map.get(location_type, 0)
        
        # Get new complex features
        temp = float(data.get('battery_temperature_c', 35.0))
        passengers = int(data.get('passenger_count', 1))
        
        # Static specs per model for capacity/weight
        model_specs = {
            'BE 6': {'capacity': 72.0, 'weight': 1600.0},
            'Creta': {'capacity': 51.0, 'weight': 1250.0},
            'Tiago EV': {'capacity': 28.0, 'weight': 1110.0},
            'Vitara': {'capacity': 58.0, 'weight': 1350.0},
            'Xpres T EV': {'capacity': 26.0, 'weight': 1105.0}
        }
        specs = model_specs.get(dataset_model_name, {'capacity': 28.0, 'weight': 1110.0})
        
        weight = float(data.get('vehicle_weight_kg', specs['weight']))
        capacity = float(data.get('battery_capacity_kwh', specs['capacity']))
        rpm = float(data.get('motor_rpm', speed_kmph * 70)) # rough estimate if not provided
        
        # Create feature DataFrame with 11 features
        feature_names = [
            'battery_percentage', 'battery_health_pct', 'current_speed_kmph',
            'odometer_km', 'model_encoded', 'location_encoded',
            'battery_temp_c', 'passenger_count', 'vehicle_weight_kg',
            'battery_capacity_kwh', 'motor_rpm'
        ]
        
        features_df = pd.DataFrame([[
            battery_percentage, battery_health, speed_kmph,
            odometer_km, model_encoded, location_encoded,
            temp, passengers, weight, capacity, rpm
        ]], columns=feature_names)
        
        # Debug logging
        print(f"\n🔍 PREDICTION (11-feature model):")
        print(f"   Model: {dataset_model_name} (encoded: {model_encoded})")
        print(f"   Location: {location_type} (encoded: {location_encoded})")
        print(f"   Battery: {battery_percentage}% | Health: {battery_health}%")
        print(f"   Speed: {speed_kmph} km/h | Odometer: {odometer_km} km")
        print(f"   Feature Vector: {features_df.values.tolist()[0]}")
        
        # Predict
        predicted_range = float(model.predict(features_df)[0])
        
        # Apply hard constraints
        predicted_range = max(0.0, predicted_range)
        
        # Cap at theoretical maximum (physics constraint)
        theoretical_max = max_range_km * (battery_percentage / 100.0) * (battery_health / 100.0)
        predicted_range = min(predicted_range, theoretical_max)
        
        print(f"   ✓ Predicted Range: {predicted_range:.2f} km (capped at {theoretical_max:.2f} km max)\n")
        
        return jsonify({
            "predicted_range_km": round(predicted_range, 2),
            "vehicle_specs_used": {
                "model": dataset_model_name,
                "max_range_km": round(max_range_km, 2)
            },
            "inputs_received": {
                "battery_percentage": battery_percentage,
                "battery_health": battery_health,
                "speed_kmph": speed_kmph,
                "odometer_km": odometer_km,
                "location_type": location_type,
                "location_encoded": location_encoded,
                "model_encoded": model_encoded
            }
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ====================================================================
# DRIVER ANALYTICS ENHANCEMENTS
# ====================================================================

@app.route('/api/admin/drivers/overspeed-comparison', methods=['GET'])
def drivers_overspeed_comparison():
    """
    Get overspeed comparison for all drivers with names
    Supports filter: ?filter=max or ?filter=min
    """
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        filter_type = request.args.get('filter', 'all')  # all, max, min
        
        # Get latest driver data
        latest_drivers = df.sort_values('date').groupby('driver_id').last().reset_index()
        
        # Calculate overspeed stats per driver
        driver_stats = []
        for driver_id in df['driver_id'].unique():
            d_df = df[df['driver_id'] == driver_id]
            
            total_overspeeds = int(d_df['overspeed_flag'].sum())
            max_speed = float(d_df['speed_kmph'].max())
            avg_speed = float(d_df[d_df['speed_kmph'] > 0]['speed_kmph'].mean())
            eco_score = max(0, min(100, 100 - (total_overspeeds * 2)))
            
            driver_stats.append({
                "driver_id": driver_id,
                "driver_name": get_driver_name(driver_id),  # ADD NAME
                "total_overspeeds": total_overspeeds,
                "max_speed": round(max_speed, 1),
                "avg_speed": round(avg_speed, 1),
                "eco_score": eco_score,
                "total_distance": round(float(d_df['trip_distance_km'].sum()), 1),
                "total_trips": int((d_df['trip_distance_km'] > 0).sum())
            })
        
        # Sort by overspeed count and add rankings
        driver_stats.sort(key=lambda x: x['total_overspeeds'])
        for idx, driver in enumerate(driver_stats, 1):
            driver['rank'] = idx
            driver['rank_label'] = '🏆' if idx <= 3 else '⚠️' if idx >= len(driver_stats) - 2 else '✓'
        
        # Apply filter
        if filter_type == 'max':
            # Show top 5 highest violators
            driver_stats = sorted(driver_stats, key=lambda x: x['total_overspeeds'], reverse=True)[:5]
        elif filter_type == 'min':
            # Show top 5 lowest violators (best performers)
            driver_stats = sorted(driver_stats, key=lambda x: x['total_overspeeds'])[:5]
        
        # Fleet statistics
        overspeed_counts = [d['total_overspeeds'] for d in driver_stats]
        
        return jsonify({
            "drivers": driver_stats,
            "fleet_avg": round(sum(overspeed_counts) / len(overspeed_counts), 1) if overspeed_counts else 0,
            "fleet_max": max(overspeed_counts) if overspeed_counts else 0,
            "fleet_min": min(overspeed_counts) if overspeed_counts else 0,
            "total_drivers": len(driver_stats),
            "filter_applied": filter_type
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/drivers/overspeed-trends', methods=['GET'])
def drivers_overspeed_trends():
    """Get overspeed trends for last N months"""
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        months_param = request.args.get('months', 3)
        months_count = int(months_param)
        
        # Get unique months sorted
        df_sorted = df.sort_values('date')
        all_months = df_sorted['month'].unique()
        
        # Get last N months
        last_n_months = list(all_months[-months_count:])
        
        # Calculate overspeed per driver per month
        drivers_data = {}
        for driver_id in df['driver_id'].unique():
            d_df = df[df['driver_id'] == driver_id]
            monthly_overspeeds = []
            
            for month in last_n_months:
                month_data = d_df[d_df['month'] == month]
                overspeed_count = int(month_data['overspeed_flag'].sum())
                monthly_overspeeds.append(overspeed_count)
            
            drivers_data[driver_id] = monthly_overspeeds
        
        return jsonify({
            "months": last_n_months,
            "drivers": drivers_data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/drivers/leaderboard', methods=['GET'])
def drivers_leaderboard():
    """Get driver leaderboard across multiple categories"""
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        leaderboard = {
            "safest": [],
            "eco_friendly": [],
            "productive": [],
            "battery_care": []
        }
        
        for driver_id in df['driver_id'].unique():
            d_df = df[df['driver_id'] == driver_id]
            latest = d_df.sort_values('date').iloc[-1]
            
            total_overspeeds = int(d_df['overspeed_flag'].sum())
            avg_efficiency = float(d_df[d_df['driving_efficiency_km_kwh'] > 0]['driving_efficiency_km_kwh'].mean())
            total_revenue = float(d_df['income'].sum())
            avg_battery_health = float(d_df['battery_health'].mean())
            
            # Safest (lowest overspeeds)
            leaderboard["safest"].append({
                "driver_id": driver_id,
                "score": total_overspeeds,
                "label": f"{total_overspeeds} incidents"
            })
            
            # Eco-friendly (best efficiency)
            leaderboard["eco_friendly"].append({
                "driver_id": driver_id,
                "score": avg_efficiency,
                "label": f"{avg_efficiency:.1f} km/kWh"
            })
            
            # Productive (highest revenue)
            leaderboard["productive"].append({
                "driver_id": driver_id,
                "score": total_revenue,
                "label": f"₹{total_revenue:,.0f}"
            })
            
            # Battery care (best health maintenance)
            leaderboard["battery_care"].append({
                "driver_id": driver_id,
                "score": avg_battery_health,
                "label": f"{avg_battery_health:.1f}%"
            })
        
        # Sort each category
        leaderboard["safest"].sort(key=lambda x: x['score'])  # Lower is better
        leaderboard["eco_friendly"].sort(key=lambda x: x['score'], reverse=True)
        leaderboard["productive"].sort(key=lambda x: x['score'], reverse=True)
        leaderboard["battery_care"].sort(key=lambda x: x['score'], reverse=True)
        
        # Get top 3 for each
        result = {
            "safest": leaderboard["safest"][:3],
            "eco_friendly": leaderboard["eco_friendly"][:3],
            "productive": leaderboard["productive"][:3],
            "battery_care": leaderboard["battery_care"][:3]
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====================================================================
# BRAND ANALYTICS ENHANCEMENTS
# ====================================================================

@app.route('/api/admin/brands/comparison', methods=['GET'])
def brands_comparison():
    """Get comparison of all brands"""
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        brands = df['brand'].unique()
        comparison_data = []
        
        for brand in brands:
            b_df = df[df['brand'] == brand]
            latest_brand = b_df.sort_values('date').groupby('vehicle_id').last()
            
            vehicle_ids = list(b_df['vehicle_id'].unique())
            vehicle_count = len(vehicle_ids)
            
            # Calculate metrics
            total_revenue = float(b_df['income'].sum())
            total_expense = float((b_df['charging_cost'] + b_df['maintenance_cost']).sum())
            profit = total_revenue - total_expense
            
            avg_efficiency = float(b_df[b_df['driving_efficiency_km_kwh'] > 0]['driving_efficiency_km_kwh'].mean())
            avg_battery_health = float(latest_brand['battery_health'].mean())
            total_distance = float(b_df['trip_distance_km'].sum())
            total_overspeeds = int(b_df['overspeed_flag'].sum())
            
            # Monthly trends
            monthly_revenue = b_df.groupby('month')['income'].sum().to_dict()
            monthly_expense = b_df.groupby('month').apply(
                lambda x: (x['charging_cost'] + x['maintenance_cost']).sum()
            ).to_dict()
            
            comparison_data.append({
                "brand": brand,
                "vehicle_count": vehicle_count,
                "vehicles": vehicle_ids,
                "total_revenue": round(total_revenue, 2),
                "total_expense": round(total_expense, 2),
                "profit": round(profit, 2),
                "profit_margin": round((profit / total_revenue * 100) if total_revenue > 0 else 0, 1),
                "avg_efficiency": round(avg_efficiency, 2),
                "avg_battery_health": round(avg_battery_health, 1),
                "total_distance": round(total_distance, 1),
                "total_overspeeds": total_overspeeds,
                "monthly_revenue": monthly_revenue,
                "monthly_expense": monthly_expense
            })
        
        # Determine best performers
        comparison_data.sort(key=lambda x: x['profit'], reverse=True)
        best_profit = comparison_data[0]['brand'] if comparison_data else None
        
        comparison_data.sort(key=lambda x: x['avg_efficiency'], reverse=True)
        most_efficient = comparison_data[0]['brand'] if comparison_data else None
        
        comparison_data.sort(key=lambda x: x['profit_margin'], reverse=True)
        best_margin = comparison_data[0]['brand'] if comparison_data else None
        
        # Sort back by brand name for display
        comparison_data.sort(key=lambda x: x['brand'])
        
        return jsonify({
            "brands": comparison_data,
            "best_profit": best_profit,
            "most_efficient": most_efficient,
            "best_margin": best_margin,
            "total_brands": len(comparison_data)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/brands/<brand_name>/vehicles', methods=['GET'])
def brand_vehicles(brand_name):
    """Get all vehicles for a specific brand"""
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        b_df = df[df['brand'].str.lower() == brand_name.lower()]
        
        if b_df.empty:
            return jsonify({"error": f"Brand {brand_name} not found"}), 404
        
        # Get latest record per vehicle
        latest = b_df.sort_values('date').groupby('vehicle_id').last().reset_index()
        
        vehicles_list = []
        for _, row in latest.iterrows():
            v_df = b_df[b_df['vehicle_id'] == row['vehicle_id']]
            
            vehicles_list.append({
                "vehicle_id": row['vehicle_id'],
                "model": row['model'],
                "body_type": row['body_type'],
                "driver_id": row['driver_id'],
                "status": row['status'],
                "battery_percentage": round(float(row['battery_percentage']), 1),
                "battery_health": round(float(row['battery_health']), 1),
                "estimated_range_km": round(float(row['estimated_range_km']), 1),
                "total_distance": round(float(v_df['trip_distance_km'].sum()), 1),
                "total_revenue": round(float(v_df['income'].sum()), 2),
                "efficiency": round(float(v_df[v_df['driving_efficiency_km_kwh'] > 0]['driving_efficiency_km_kwh'].mean()), 2)
            })
        
        return jsonify({
            "brand": brand_name,
            "vehicle_count": len(vehicles_list),
            "vehicles": vehicles_list
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/fleet/monthly-comparison', methods=['GET'])
def fleet_monthly_comparison():
    """Compare two months performance"""
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        month1 = request.args.get('month1', 'May')
        month2 = request.args.get('month2', 'June')
        
        m1_df = df[df['month'] == month1]
        m2_df = df[df['month'] == month2]
        
        def get_month_stats(month_df):
            return {
                "revenue": float(month_df['income'].sum()),
                "expense": float((month_df['charging_cost'] + month_df['maintenance_cost']).sum()),
                "distance": float(month_df['trip_distance_km'].sum()),
                "energy": float(month_df['energy_consumed_kwh'].sum()),
                "overspeeds": int(month_df['overspeed_flag'].sum()),
                "trips": int((month_df['trip_distance_km'] > 0).sum())
            }
        
        stats1 = get_month_stats(m1_df) if not m1_df.empty else {}
        stats2 = get_month_stats(m2_df) if not m2_df.empty else {}
        
        # Calculate changes
        changes = {}
        for key in stats1.keys():
            if key in stats2 and stats1.get(key, 0) != 0:
                change_pct = ((stats2[key] - stats1[key]) / stats1[key]) * 100
                changes[key] = {
                    "value": round(stats2[key] - stats1[key], 2),
                    "percentage": round(change_pct, 1),
                    "trend": "up" if change_pct > 0 else "down"
                }
        
        return jsonify({
            "month1": {"name": month1, "stats": stats1},
            "month2": {"name": month2, "stats": stats2},
            "changes": changes
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# DRIVER MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route('/api/admin/driver-management/list', methods=['GET'])
def get_all_driver_details():
    """Get all drivers with their details from driver_details table"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT driver_id, driver_name, license_number, email, phone, 
                   assigned_vehicle_id, base_salary_per_km, created_date
            FROM driver_details
            ORDER BY driver_id
        ''')
        rows = c.fetchall()
        conn.close()
        
        drivers = []
        for row in rows:
            drivers.append({
                "driver_id": row[0],
                "driver_name": row[1],
                "license_number": row[2],
                "email": row[3],
                "phone": row[4],
                "assigned_vehicle_id": row[5],
                "base_salary_per_km": row[6],
                "created_date": row[7]
            })
        
        return jsonify(drivers), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/driver-management/add', methods=['POST'])
def add_driver_details():
    """Add a new driver with details"""
    try:
        data = request.json
        driver_id = data.get('driver_id')
        driver_name = data.get('driver_name')
        license_number = data.get('license_number')
        email = data.get('email', '')
        phone = data.get('phone', '')
        assigned_vehicle_id = data.get('assigned_vehicle_id', '')
        base_salary_per_km = float(data.get('base_salary_per_km', 8.0))
        
        from datetime import datetime
        created_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO driver_details 
            (driver_id, driver_name, license_number, email, phone, 
             assigned_vehicle_id, base_salary_per_km, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (driver_id, driver_name, license_number, email, phone, 
              assigned_vehicle_id, base_salary_per_km, created_date))
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Driver added successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Driver ID already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/driver-management/update/<driver_id>', methods=['PUT'])
def update_driver_details(driver_id):
    """Update driver details"""
    try:
        data = request.json
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Build dynamic update query
        update_fields = []
        values = []
        
        if 'driver_name' in data:
            update_fields.append('driver_name = ?')
            values.append(data['driver_name'])
        if 'license_number' in data:
            update_fields.append('license_number = ?')
            values.append(data['license_number'])
        if 'email' in data:
            update_fields.append('email = ?')
            values.append(data['email'])
        if 'phone' in data:
            update_fields.append('phone = ?')
            values.append(data['phone'])
        if 'assigned_vehicle_id' in data:
            update_fields.append('assigned_vehicle_id = ?')
            values.append(data['assigned_vehicle_id'])
        if 'base_salary_per_km' in data:
            update_fields.append('base_salary_per_km = ?')
            values.append(float(data['base_salary_per_km']))
        
        if not update_fields:
            return jsonify({"error": "No fields to update"}), 400
        
        values.append(driver_id)
        query = f"UPDATE driver_details SET {', '.join(update_fields)} WHERE driver_id = ?"
        
        c.execute(query, values)
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Driver updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/driver-management/delete/<driver_id>', methods=['DELETE'])
def delete_driver_details(driver_id):
    """Delete a driver"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM driver_details WHERE driver_id = ?', (driver_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Driver deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/driver-management/assign-trip', methods=['POST'])
def assign_driver_trip():
    """Assign a trip to a driver and calculate salary"""
    try:
        data = request.json
        driver_id = data.get('driver_id')
        vehicle_id = data.get('vehicle_id', '')
        trip_date = data.get('trip_date')
        destination_km = float(data.get('destination_km'))
        
        # Fixed salary rate: ₹2.30 per km (₹800 for 350km)
        SALARY_RATE_PER_KM = 2.30
        salary_earned = destination_km * SALARY_RATE_PER_KM
        
        from datetime import datetime
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Insert into driver_trips table
        c.execute('''
            INSERT INTO driver_trips 
            (driver_id, vehicle_id, trip_date, destination_km, salary_earned, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (driver_id, vehicle_id, trip_date, destination_km, salary_earned, 'Pending', created_at))
        
        # Insert into trip_salaries table
        c.execute('''
            INSERT INTO trip_salaries 
            (driver_id, trip_date, destination_km, salary_amount)
            VALUES (?, ?, ?, ?)
        ''', (driver_id, trip_date, destination_km, salary_earned))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Trip assigned successfully",
            "salary_earned": round(salary_earned, 2),
            "salary_rate": SALARY_RATE_PER_KM
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/driver-management/trips/<driver_id>', methods=['GET'])
def get_driver_trips(driver_id):
    """Get all trips for a specific driver"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT trip_assignment_id, driver_id, vehicle_id, trip_date, 
                   destination_km, salary_earned, status, created_at
            FROM driver_trips
            WHERE driver_id = ?
            ORDER BY trip_date DESC
        ''', (driver_id,))
        rows = c.fetchall()
        conn.close()
        
        trips = []
        for row in rows:
            trips.append({
                "trip_assignment_id": row[0],
                "driver_id": row[1],
                "vehicle_id": row[2],
                "trip_date": row[3],
                "destination_km": row[4],
                "salary_earned": row[5],
                "status": row[6],
                "created_at": row[7]
            })
        
        return jsonify(trips), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/driver/salary-details/<driver_id>', methods=['GET'])
def get_driver_salary_details(driver_id):
    """Get driver's salary summary for driver dashboard"""
    try:
        # Resolve email to driver ID if needed
        if '@' in driver_id:
            driver_id = get_driver_id_from_email(driver_id)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get driver details
        c.execute('''
            SELECT driver_name, base_salary_per_km, assigned_vehicle_id
            FROM driver_details
            WHERE driver_id = ?
        ''', (driver_id,))
        driver_row = c.fetchone()
        
        if not driver_row:
            conn.close()
            return jsonify({
                "total_trips": 0,
                "total_distance_km": 0.0,
                "total_salary_earned": 0.0,
                "average_trip_distance": 0.0,
                "recent_trips": []
            }), 200
        
        driver_name = driver_row[0]
        base_salary_per_km = driver_row[1]
        assigned_vehicle = driver_row[2]
        
        # Get salary from trip_salaries table (realistic calculation)
        c.execute('''
            SELECT COUNT(*), SUM(destination_km), SUM(salary_amount)
            FROM trip_salaries
            WHERE driver_id = ?
        ''', (driver_id,))
        stats_row = c.fetchone()
        
        total_trips = stats_row[0] if stats_row[0] else 0
        total_distance = stats_row[1] if stats_row[1] else 0.0
        total_salary = stats_row[2] if stats_row[2] else 0.0
        avg_trip_distance = total_distance / total_trips if total_trips > 0 else 0.0
        
        # Get recent trips (last 10) from trip_salaries
        c.execute('''
            SELECT ts.trip_date, dt.vehicle_id, ts.destination_km, ts.salary_amount, dt.status
            FROM trip_salaries ts
            LEFT JOIN driver_trips dt ON ts.driver_id = dt.driver_id AND ts.trip_date = dt.trip_date
            WHERE ts.driver_id = ?
            ORDER BY ts.trip_date DESC
            LIMIT 10
        ''', (driver_id,))
        recent_rows = c.fetchall()
        
        conn.close()
        
        recent_trips = []
        for row in recent_rows:
            recent_trips.append({
                "trip_date": row[0],
                "vehicle_id": row[1] or '--',
                "destination_km": row[2],
                "salary_earned": row[3],
                "status": row[4] or 'Completed'
            })
        
        return jsonify({
            "driver_name": driver_name,
            "base_salary_per_km": base_salary_per_km,
            "assigned_vehicle": assigned_vehicle,
            "total_trips": total_trips,
            "total_distance_km": round(total_distance, 2),
            "total_salary_earned": round(total_salary, 2),
            "average_trip_distance": round(avg_trip_distance, 2),
            "recent_trips": recent_trips
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)

