import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# VEHICLE SPECS
# ─────────────────────────────────────────────────────────────
VEHICLE_SPECS = {
    "Xpres T EV": {
        "brand": "Tata",     "body_type": "Sedan",    "total_seats": 4,
        "battery_capacity": 26,  "vehicle_weight": 1105,
        "max_speed": 120,        "max_range": 395,      "motor_rpm": 9500,
        "income_per_km": 22,     "base_kwh_per_km": 26 / 395,
    },
    "Tiago EV": {
        "brand": "Tata",     "body_type": "Hatchback", "total_seats": 5,
        "battery_capacity": 28,  "vehicle_weight": 1110,
        "max_speed": 120,        "max_range": 405,      "motor_rpm": 9700,
        "income_per_km": 25,     "base_kwh_per_km": 28 / 405,
    },
    "BE 6": {
        "brand": "Mahindra", "body_type": "SUV",       "total_seats": 7,
        "battery_capacity": 72,  "vehicle_weight": 1600,
        "max_speed": 200,        "max_range": 430,      "motor_rpm": 10500,
        "income_per_km": 30,     "base_kwh_per_km": 72 / 430,
    },
    "Vitara": {
        "brand": "Suzuki",   "body_type": "SUV",       "total_seats": 7,
        "battery_capacity": 58,  "vehicle_weight": 1350,
        "max_speed": 150,        "max_range": 420,      "motor_rpm": 10200,
        "income_per_km": 30,     "base_kwh_per_km": 58 / 420,
    },
    "Creta": {
        "brand": "Hyundai",  "body_type": "SUV",       "total_seats": 7,
        "battery_capacity": 51,  "vehicle_weight": 1250,
        "max_speed": 130,        "max_range": 410,      "motor_rpm": 9900,
        "income_per_km": 30,     "base_kwh_per_km": 51 / 410,
    },
}

VEHICLE_IDS = [f"E{i:03d}" for i in range(1, 21)]
POOL_CARS   = set()
VEHICLE_MODEL = {}
for i, vid in enumerate(VEHICLE_IDS):
    if   i < 6:  VEHICLE_MODEL[vid] = "Xpres T EV"
    elif i < 10: VEHICLE_MODEL[vid] = "Tiago EV"
    elif i < 13: VEHICLE_MODEL[vid] = "BE 6"
    elif i < 17: VEHICLE_MODEL[vid] = "Vitara"
    else:         VEHICLE_MODEL[vid] = "Creta"

VEHICLE_DRIVER = {f"E{i:03d}": f"D{i:03d}" for i in range(1, 21)}
ALL_DRIVERS    = list(VEHICLE_DRIVER.values())

INCOME_MULTIPLIER = {
    "running": 1.00, "charging": 0.65, "garage": 0.45, "workshop": 0.35,
}

CHARGE_COST_PER_KWH  = 20.0
FORCED_WS_DAYS       = 6
FORCED_WS_COST       = 300.0
HEALTH_FORCED_WS     = 50.0
HEALTH_HALF_RANGE    = 60.0

MAINT = {
    "K": 50.0, "alpha": 1.50, "beta": 6.00, 
    "gamma": 0.009, "delta": 8.0, "lam": 1.50
}

START_DATE = datetime(2026, 3, 1,  0, 0, 0)
END_DATE   = datetime(2026, 6, 10, 23, 30, 0)
HALF_HOUR  = timedelta(minutes=30)

def degrade_battery_health(charge_count, days_elapsed, high_temp_events, overspeed_events):
    cycle_wear    = (charge_count / 500.0) * 10.0
    cal_decay     = (days_elapsed / 365.25) * 3.0
    temp_penalty  = high_temp_events * 0.02
    speed_stress  = (overspeed_events / 200.0) * 2.0
    noise         = random.gauss(0.0, 1.5)
    health = 100.0 - cycle_wear - cal_decay - temp_penalty - speed_stress + noise
    return round(min(100.0, max(0, health)), 4)

def compute_energy_rate(spec, health_ratio, passenger_count, speed_ratio):
    rate = (
        spec["base_kwh_per_km"]
        * (1 / max(health_ratio, 0.01))
        * (1 + (passenger_count - 1) * 0.05)
        * (1 + speed_ratio * 0.35)
        * random.gauss(1.0, 0.008)
    )
    return max(rate, 0.05)

def assign_location(trip_distance_km):
    if trip_distance_km < 50.0:
        return "highway" if random.random() < 0.05 else "city"
    else:
        return "city"    if random.random() < 0.05 else "highway"

def make_forced_ws_row(row_id, timestamp, vid, spec, model, state, health_ratio, days_elapsed):
    battery_health = state["battery_health"]
    max_range_km   = round(spec["max_range"] * health_ratio / 2.0, 2)
    motor_rpm_cur  = round(spec["motor_rpm"] * health_ratio)
    est_range      = spec["max_range"] * random.uniform(0.1, 0.5) 
    bpct           = int(random.uniform(5, 50))
    
    return {
        "trip_id":                 f"T{row_id:06d}",
        "date_time":               timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "vehicle_id":              vid,
        "driver_id":               "WORKSHOP",
        "brand":                   spec["brand"],
        "model":                   model,
        "body_type":               spec["body_type"],
        "total_seats":             spec["total_seats"],
        "battery_capacity_kwh":    spec["battery_capacity"],
        "vehicle_weight_kg":       spec["vehicle_weight"],
        "max_speed_kmph":          spec["max_speed"],
        "status":                  "workshop",
        "location":                "city",
        "trip_distance_km":        0.0,
        "passenger_count":         0,
        "battery_percentage":      bpct,
        "charging_status":         0,
        "battery_temp_c":          round(random.uniform(28, 40), 1),
        "charge_count":            state["charge_count"],
        "battery_health_pct":      battery_health,
        "max_range_km":            max_range_km,
        "motor_rpm":               int(motor_rpm_cur),
        "current_max_speed_kmph":  0.0,
        "current_speed_kmph":      0.0,
        "overspeeding_flag":       0,
        "energy_consumed_kwh":     0.0,
        "driving_efficiency_pct":  0.0,
        "estimated_range_km":      round(est_range, 2),
        "ride_complete":           state["ride_complete"],
        "workshop_visit_count":    state["workshop_visit_count"],
        "income_inr":              0.0,
        "maintenance_cost_inr":    FORCED_WS_COST * random.uniform(0.8, 1.5),
        "charge_cost_inr":         0.0,
        "is_forced_workshop":      1,
    }

def pick_status(hour):
    if hour < 6 or hour >= 22:
        return random.choices(
            ["running", "charging", "workshop", "garage"], 
            weights=[0.20, 0.30, 0.05, 0.45]
        )[0]
    elif (7 <= hour < 10) or (17 <= hour < 20):
        return random.choices(
            ["running", "charging", "workshop", "garage"], 
            weights=[0.72, 0.12, 0.06, 0.10]
        )[0]
    else:
        return random.choices(
            ["running", "charging", "workshop", "garage"], 
            weights=[0.60, 0.20, 0.08, 0.12]
        )[0]

def pick_trip_distance():
    r = random.random()
    if r < 0.60: return round(random.uniform(15, 60), 2)
    elif r < 0.90: return round(random.uniform(40, 150), 2)
    else: return round(random.uniform(100, 300), 2)

def generate_ev_dataset() -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    
    states = {
        vid: {
            "battery_health":       100.0,
            "charge_count":         0,
            "ride_complete":        0,
            "workshop_visit_count": 0,
            "high_temp_events":     0,
            "overspeed_events":     0,
            "forced_workshop":      False,
            "workshop_slots_left":  0,
            "current_battery_pct":  100.0,
            "active_task":          "garage",
            "task_slots_left":      0,
            "current_location":     "garage",
            "task_trip_distance_per_slot": 0.0,
            "task_speed_target":    0.0,
        }
        for vid in VEHICLE_IDS
    }
    
    rows       = []
    row_id     = 1
    slot_time  = START_DATE
    
    while slot_time <= END_DATE:
        days_elapsed = (slot_time - START_DATE).days
        hour         = slot_time.hour
        
        for vid in VEHICLE_IDS:
            spec   = VEHICLE_SPECS[VEHICLE_MODEL[vid]]
            model  = VEHICLE_MODEL[vid]
            state  = states[vid]
            
            ts = slot_time + timedelta(seconds=random.randint(0, 1799))
            
            if state["forced_workshop"]:
                state["workshop_slots_left"] -= 1
                if state["workshop_slots_left"] <= 0:
                    state["forced_workshop"]  = False
                    state["battery_health"]   = 100.0
                    state["charge_count"]     = 0
                    state["current_battery_pct"] = 100.0
                
                rows.append(make_forced_ws_row(
                    row_id, ts, vid, spec, model, state, 
                    state["battery_health"] / 100.0, days_elapsed
                ))
                row_id += 1
                continue
            
            if state["task_slots_left"] <= 0:
                new_status = pick_status(hour)
                if new_status == "running" and state["current_battery_pct"] < 15.0:
                    new_status = "charging"
                
                state["active_task"] = new_status
                
                if new_status == "running":
                    total_dist = pick_trip_distance()
                    avg_speed = random.uniform(30, 80)
                    total_hours = total_dist / avg_speed
                    slots = max(1, int(round(total_hours * 2)))
                    state["task_slots_left"] = slots
                    state["task_trip_distance_per_slot"] = total_dist / slots
                    state["task_speed_target"] = avg_speed
                    state["current_location"] = assign_location(total_dist)
                elif new_status == "charging":
                    slots = random.choice([2, 4, 10, 14])
                    state["task_slots_left"] = slots
                    state["current_location"] = "garage"
                elif new_status == "workshop":
                    state["task_slots_left"] = random.randint(2, 6)
                    state["current_location"] = "garage"
                    state["workshop_visit_count"] += 1
                else:
                    state["task_slots_left"] = random.randint(1, 4)
                    state["current_location"] = "garage"
            
            state["task_slots_left"] -= 1
            
            status = state["active_task"]
            driver = VEHICLE_DRIVER[vid]
            location = state["current_location"]
            
            if status == "running" and state["current_battery_pct"] <= 2.0:
                status = "garage"
                state["active_task"] = "garage"
                state["task_slots_left"] = 4
                state["current_location"] = "city"
            
            passenger_count = random.randint(1, spec["total_seats"])
            charging_status = 1 if status == "charging" else 0
            
            battery_health = degrade_battery_health(
                state["charge_count"], days_elapsed, 
                state["high_temp_events"], state["overspeed_events"]
            )
            state["battery_health"] = battery_health
            health_ratio = battery_health / 100.0
            
            if battery_health < HEALTH_FORCED_WS and not state["forced_workshop"]:
                state["forced_workshop"]      = True
                state["workshop_slots_left"]  = FORCED_WS_DAYS * 48
                state["workshop_visit_count"] += 1
                rows.append(make_forced_ws_row(
                    row_id, ts, vid, spec, model, state, health_ratio, days_elapsed
                ))
                row_id += 1
                continue
            
            max_range_km  = round(spec["max_range"] * health_ratio, 2)
            if battery_health < HEALTH_HALF_RANGE: 
                max_range_km = round(max_range_km / 2.0, 2)
            motor_rpm_cur = round(spec["motor_rpm"] * health_ratio)
            
            energy_consumed = 0.0
            charge_cost = 0.0
            
            if status == "running":
                trip_distance = state["task_trip_distance_per_slot"] * random.uniform(0.85, 1.15)
                current_speed = state["task_speed_target"] * random.uniform(0.85, 1.15)
                cur_max_speed = max(20.0, spec["max_speed"] * health_ratio)
                current_speed = min(current_speed, cur_max_speed)
                speed_ratio = current_speed / max(spec["max_speed"], 1)
                
                energy_rate = compute_energy_rate(spec, health_ratio, passenger_count, speed_ratio)
                energy_rate *= random.uniform(0.6, 1.8) 
                energy_consumed = round(energy_rate * trip_distance, 4)
                
                # Speed > 80 kmph: battery drains 1.2x faster
                speed_drain_multiplier = 1.2 if current_speed > 80.0 else 1.0
                pct_drained = (energy_consumed / spec["battery_capacity"]) * 100 * speed_drain_multiplier
                state["current_battery_pct"] = max(1.5, state["current_battery_pct"] - pct_drained)
                state["ride_complete"] += 1
            else:
                trip_distance = 0.0
                current_speed = 0.0
                cur_max_speed = 0.0
                
                if charging_status:
                    charge_rate_kw = 7.2 if random.random() < 0.6 else 50.0
                    energy_added = charge_rate_kw * 0.5
                    pct_added = (energy_added / spec["battery_capacity"]) * 100
                    kwh_charged_this_slot = min(
                        energy_added, 
                        spec["battery_capacity"] * (1 - state["current_battery_pct"]/100)
                    )
                    charge_cost = round(kwh_charged_this_slot * CHARGE_COST_PER_KWH, 2)
                    
                    if state["current_battery_pct"] + pct_added >= 100.0:
                        state["current_battery_pct"] = 100.0
                        state["charge_count"] += 1
                        state["task_slots_left"] = 0
                    else:
                        state["current_battery_pct"] += pct_added
            
            overspeeding_flag = 1 if current_speed > 100 else 0
            if overspeeding_flag: 
                state["overspeed_events"] += 1
            
            if status in ("garage", "workshop"): 
                battery_temp = round(random.uniform(18, 24), 1)
            elif status == "running":
                base_temp = 18 + (1 - health_ratio) * 20
                battery_temp = round(max(15, min(45, random.gauss(base_temp, 1.0))), 1)
            else:
                battery_temp = round(random.uniform(15, 42), 1)
            
            if battery_temp > 35: 
                state["high_temp_events"] += 1
            
            # estimated range: battery charge × health (linear) × noise
            # health_ratio directly scales range — lower health = proportionally shorter range
            base_range = max_range_km * (state["current_battery_pct"] / 100.0) * health_ratio
            estimated_range_km = base_range * random.uniform(0.65, 1.35)
            estimated_range_km = round(max(0.0, estimated_range_km), 2)
            
            optimal_energy = spec["base_kwh_per_km"] * trip_distance
            driving_eff = round(min(100.0, (optimal_energy / max(energy_consumed, 1e-6)) * 100), 2)
            if status != "running": 
                driving_eff = 0.0
            
            income = trip_distance * spec["income_per_km"] * INCOME_MULTIPLIER[status]
            income *= random.uniform(0.5, 1.5)
            income = round(income, 2)
            
            R_thousands = motor_rpm_cur / 1000
            maint_cost  = round(
                MAINT["K"] + MAINT["alpha"] * trip_distance + MAINT["beta"] * energy_consumed
                + MAINT["gamma"] * R_thousands + MAINT["delta"] * passenger_count 
                - MAINT["lam"] * driving_eff, 2
            )
            maint_cost *= random.uniform(0.3, 2.5)
            maint_cost = max(0.0, maint_cost)
            
            rows.append({
                "trip_id":                 f"T{row_id:06d}",
                "date_time":               ts.strftime("%Y-%m-%d %H:%M:%S"),
                "vehicle_id":              vid,
                "driver_id":               driver,
                "brand":                   spec["brand"],
                "model":                   model,
                "body_type":               spec["body_type"],
                "total_seats":             spec["total_seats"],
                "battery_capacity_kwh":    spec["battery_capacity"],
                "vehicle_weight_kg":       spec["vehicle_weight"],
                "max_speed_kmph":          spec["max_speed"],
                "status":                  status,
                "location":                location,
                "trip_distance_km":        round(trip_distance, 2),
                "passenger_count":         passenger_count,
                "battery_percentage":      int(state["current_battery_pct"]),
                "charging_status":         charging_status,
                "battery_temp_c":          battery_temp,
                "charge_count":            state["charge_count"],
                "battery_health_pct":      battery_health,
                "max_range_km":            max_range_km,
                "motor_rpm":               int(motor_rpm_cur),
                "current_max_speed_kmph":  round(cur_max_speed, 2),
                "current_speed_kmph":      round(current_speed, 2),
                "overspeeding_flag":       overspeeding_flag,
                "energy_consumed_kwh":     energy_consumed,
                "driving_efficiency_pct":  driving_eff,
                "estimated_range_km":      estimated_range_km,
                "ride_complete":           state["ride_complete"],
                "workshop_visit_count":    state["workshop_visit_count"],
                "income_inr":              income,
                "maintenance_cost_inr":    round(maint_cost, 2),
                "charge_cost_inr":         charge_cost,
                "is_forced_workshop":      0,
            })
            row_id += 1
        
        slot_time += HALF_HOUR
    
    df = pd.DataFrame(rows)
    df = df.sort_values("date_time").reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────
# GENERATE 20K DEAD-BATTERY ROWS
# battery_percentage == 0  →  estimated_range_km == 0.0
# ─────────────────────────────────────────────────────────────
def generate_dead_battery_rows(existing_df: pd.DataFrame, n: int = 20000) -> pd.DataFrame:
    """Synthesise `n` rows that represent vehicles with a fully depleted battery.
    Rules:
    - battery_percentage      = 0
    - estimated_range_km      = 0.0
    - status                  = 'garage' (stranded / waiting for tow/charge)
    - trip_distance_km        = 0.0
    - current_speed_kmph      = 0.0
    - current_max_speed_kmph  = 0.0
    - charging_status         = 0  (not yet plugged in)
    - income_inr              = 0.0
    - All other fields are sampled realistically from the existing dataset
      so the rows blend naturally.
    """
    random.seed(99)
    np.random.seed(99)
    
    # Pool of timestamps spread across the simulation window
    total_seconds = int((END_DATE - START_DATE).total_seconds())
    
    # Reference columns from the existing data for realistic sampling
    ref = existing_df[existing_df["status"].isin(["garage", "running"])].copy()
    
    dead_rows = []
    all_vids = VEHICLE_IDS
    
    for i in range(n):
        vid   = random.choice(all_vids)
        model = VEHICLE_MODEL[vid]
        spec  = VEHICLE_SPECS[model]
        
        # Random timestamp within simulation window
        rand_secs = random.randint(0, total_seconds)
        ts = START_DATE + timedelta(seconds=rand_secs)
        
        # Sample a realistic battery health from existing data for this vehicle
        vehicle_ref = ref[ref["vehicle_id"] == vid]
        if len(vehicle_ref) > 0:
            bh = float(vehicle_ref["battery_health_pct"].sample(1).values[0])
        else:
            bh = round(random.uniform(55.0, 95.0), 4)
        
        health_ratio  = bh / 100.0
        max_range_km  = round(spec["max_range"] * health_ratio, 2)
        motor_rpm_cur = round(spec["motor_rpm"] * health_ratio)
        
        # Charge count & workshop visits — sample from existing vehicle data
        if len(vehicle_ref) > 0:
            charge_count   = int(vehicle_ref["charge_count"].sample(1).values[0])
            ws_count       = int(vehicle_ref["workshop_visit_count"].sample(1).values[0])
            ride_complete  = int(vehicle_ref["ride_complete"].sample(1).values[0])
        else:
            charge_count  = random.randint(0, 30)
            ws_count      = random.randint(0, 5)
            ride_complete = random.randint(0, 200)
        
        battery_temp = round(random.uniform(18, 30), 1)  # cool — not running
        
        # Maintenance cost still applies (stranded vehicle may need roadside help)
        maint_cost = round(
            (MAINT["K"] + MAINT["delta"] * random.randint(1, spec["total_seats"]))
            * random.uniform(0.5, 2.0), 2
        )
        
        dead_rows.append({
            "trip_id":                 f"DB{i+1:06d}",          # DB = Dead Battery
            "date_time":               ts.strftime("%Y-%m-%d %H:%M:%S"),
            "vehicle_id":              vid,
            "driver_id":               VEHICLE_DRIVER[vid],
            "brand":                   spec["brand"],
            "model":                   model,
            "body_type":               spec["body_type"],
            "total_seats":             spec["total_seats"],
            "battery_capacity_kwh":    spec["battery_capacity"],
            "vehicle_weight_kg":       spec["vehicle_weight"],
            "max_speed_kmph":          spec["max_speed"],
            "status":                  "garage",
            "location":                random.choice(["city", "highway"]),
            "trip_distance_km":        0.0,
            "passenger_count":         0,
            "battery_percentage":      0,               # ← KEY: battery = 0
            "charging_status":         0,
            "battery_temp_c":          battery_temp,
            "charge_count":            charge_count,
            "battery_health_pct":      bh,
            "max_range_km":            max_range_km,
            "motor_rpm":               int(motor_rpm_cur),
            "current_max_speed_kmph":  0.0,
            "current_speed_kmph":      0.0,
            "overspeeding_flag":       0,
            "energy_consumed_kwh":     0.0,
            "driving_efficiency_pct":  0.0,
            "estimated_range_km":      0.0,             # ← KEY: range = 0
            "ride_complete":           ride_complete,
            "workshop_visit_count":    ws_count,
            "income_inr":              0.0,
            "maintenance_cost_inr":    maint_cost,
            "charge_cost_inr":         0.0,
            "is_forced_workshop":      0,
        })
    
    return pd.DataFrame(dead_rows)

def print_report(df):
    normal  = df[df["is_forced_workshop"] == 0]
    numeric = normal.select_dtypes(include=[float, int])
    corr    = numeric.corr()
    
    print(f"\n{'='*64}\n  Correlations with estimated_range_km  (normal rows)\n{'='*64}")
    ser = corr["estimated_range_km"].drop("estimated_range_km").sort_values(key=abs, ascending=False)
    for feat, val in ser.head(12).items():
        print(f"  {feat:<35s}  {val:+.3f}  {'#'*int(abs(val)*20)}")

if __name__ == "__main__":
    print("Generating EV fleet dataset v12 (Stateful + Noise + Dead-Battery + Speed/Health range) …")
    
    # ── Step 1: original dataset ──────────────────────────────────────────────
    df_main = generate_ev_dataset()
    print(f"Base dataset:        {len(df_main):,} rows × {df_main.shape[1]} columns")
    
    # ── Step 2: 20 000 dead-battery rows ─────────────────────────────────────
    print("Generating 20,000 dead-battery rows (battery_pct=0 → estimated_range=0) …")
    df_dead = generate_dead_battery_rows(df_main, n=20_000)
    print(f"Dead-battery rows:   {len(df_dead):,} rows")
    
    # ── Step 3: combine & sort ────────────────────────────────────────────────
    df_final = pd.concat([df_main, df_dead], ignore_index=True)
    df_final = df_final.sort_values("date_time").reset_index(drop=True)
    print(f"Final dataset:       {len(df_final):,} rows × {df_final.shape[1]} columns")
    
    # ── Sanity check ──────────────────────────────────────────────────────────
    dead_check = df_final[df_final["battery_percentage"] == 0]
    assert (dead_check["estimated_range_km"] == 0.0).all(), \
        "FAIL: some battery=0 rows have non-zero estimated_range_km!"
    print(f"\n✓ Sanity check passed: all {len(dead_check):,} rows with battery_percentage=0 "
          f"have estimated_range_km=0.0")
    
    print_report(df_final)
    
    out_csv = "nev_fleet_dataset_v16_odo.csv"
    print(f"\nSaving CSV → {out_csv} …")
    df_final.to_csv(out_csv, index=False)
    print("Done.")
