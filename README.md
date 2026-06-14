# EV Fleet Management System & Range Predictor 🏎️⚡

An advanced, web-based Electric Vehicle (EV) Fleet Management Dashboard integrating a Machine Learning model to accurately predict the remaining driving range of vehicles based on dynamic telemetry and physics-based parameters.

## 🌟 Project Overview

This project provides a comprehensive solution for managing a fleet of electric vehicles. It features role-based access (Driver vs. Admin), real-time monitoring of battery health and performance, charging metrics, and an integrated AI model that acts as a digital twin to predict EV ranges more accurately than static factory estimates. 

The application is accompanied by a stunning 3D-animated login portal utilizing Three.js, setting a premium tone for the user experience.

---

## 🏗️ System Architecture

The project follows a modular, monolithic architecture:

1. **Frontend (UI/UX)**
   * **`index.html`**: A cinematic, 3D animated login screen built with **Three.js** rendering a high-fidelity 3D model (`2011_corvette_z06_carbon_limited_edition_nfs.glb`).
   * **`dashboard.html`**: The primary operational dashboard handling all REST API communications for telemetry and user data.
   * *Stack*: Vanilla HTML, CSS, JavaScript.

2. **Backend (API & Logic)**
   * **`backend.py`**: A **Flask**-based REST API that serves the frontend, processes driver/admin authentication via a **SQLite** database (`users.db`), loads the synthetic EV dataset into memory, and hosts the active Machine Learning model to provide live range estimates.

3. **Machine Learning Pipeline**
   * Built with **Python, Pandas, and scikit-learn**.
   * Responsible for ingesting synthetic EV telemetry, performing feature engineering, and training a `RandomForestRegressor` to map complex environmental and mechanical variables to remaining vehicle range.

---

## 📊 Data Generation

Because real-world, high-fidelity EV telemetry data is often proprietary, this project includes a robust data generator (`generate_dataset.py`).

*   **Stateful Simulation:** The generator produces highly realistic, stateful trips, ensuring continuity in odometer readings and battery discharge cycles across multiple drivers.
*   **Physics-Based Variables:** It mathematically models the relationship between speed, battery temperature, vehicle weight, motor RPM, and battery capacity.
*   **Output:** The script generates `nev_fleet_dataset_v16_odo.csv`, which serves as both the training ground for the ML model and the simulated live database for the backend dashboard.

---

## 🧠 Model Training

The core intelligence of the application resides in **`train_complex_model.py`**.

1.  **Feature Engineering:** The script reads the synthetic dataset, computes cumulative metrics (like `odometer_km`), and encodes categorical data (location, vehicle model).
2.  **Algorithm:** We use a **Random Forest Regressor** (`RandomForestRegressor(n_estimators=100)`) due to its excellent performance on non-linear tabular data and complex interaction effects (e.g., how high speeds *and* high temperatures degrade range).
3.  **Features Used:** 11 comprehensive features including `battery_percentage`, `battery_health_pct`, `current_speed_kmph`, `battery_temp_c`, `passenger_count`, `motor_rpm`, and more.
4.  **Artifact Generation:** Once trained, the model is serialized into `estimated_range_model_complex.pkl` for immediate consumption by the Flask backend.

> ⚠️ **IMPORTANT NOTE ON THE MODEL FILE (`.pkl`)**
> The resulting `estimated_range_model_complex.pkl` file is approximately 480MB. Because GitHub enforces a strict 100MB file limit, **this `.pkl` file is excluded from version control via `.gitignore`.** 
> 
> *Anyone cloning this repository must manually run the training script to generate the model locally before starting the backend.*

---

## 🚀 Getting Started

Follow these steps to run the application on your local machine:

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install flask flask-cors pandas numpy scikit-learn
```

### 2. Generate the AI Model
Because the `.pkl` model file is too large for GitHub, you must build it locally first:
```bash
python train_complex_model.py
```
*This will read `nev_fleet_dataset_v16_odo.csv` and generate `estimated_range_model_complex.pkl`.*

### 3. Start the Backend Server
Once the model is ready, boot up the Flask server:
```bash
python backend.py
```

### 4. Access the Dashboard
Open your web browser and navigate to:
`http://localhost:5000`

Login using the Driver or Admin credentials provided in your SQLite database. Enjoy the drift!
