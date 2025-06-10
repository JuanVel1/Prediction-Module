import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from supabase import create_client
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

class SupabaseTrainer:
    def __init__(self, table_name="weather_training_data"):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_ANON_KEY") 
        self.table_name = table_name
        self.timestamp_file = "last_check.json"
        
        if self.url and self.key:
            self.client = create_client(self.url, self.key)
        else:
            self.client = None
    
    def get_last_timestamp(self):
        try:
            with open(self.timestamp_file, 'r') as f:
                return json.load(f).get('last_check')
        except:
            return None
    
    def update_timestamp(self, timestamp):
        with open(self.timestamp_file, 'w') as f:
            json.dump({'last_check': timestamp}, f)
    
    def check_new_data(self):
        if not self.client:
            return False, None
        
        last_check = self.get_last_timestamp()
        if not last_check:
            last_check = (datetime.utcnow() - timedelta(days=1)).isoformat()
        
        response = self.client.table(self.table_name).select("*").gt("created_at", last_check).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            latest = max(row['created_at'] for row in response.data)
            self.update_timestamp(latest)
            return True, df
        return False, None
    
    def prepare_data(self, df):
        features = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 'precipitation', 
                   'surface_pressure', 'cloudcover', 'windspeed_10m', 'windgusts_10m']
        
        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0).values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        y_pressure = df['surface_pressure'].fillna(df['surface_pressure'].mean()).values
        y_precip = df['precipitation'].fillna(0).values
        
        return X_scaled, y_pressure, y_precip
    
    def train_lstm(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(1, X_train.shape[2])),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        model.save("models/surface_pressure.keras")
        
        return {"mse": float(tf.keras.losses.mse(y_test, model.predict(X_test, verbose=0)).numpy().mean())}
    
    def train_xgb(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        with open("models/precipitation.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        return {"mse": float(((y_test - model.predict(X_test)) ** 2).mean())}
    
    def retrain(self):
        has_new, data = self.check_new_data()
        if not has_new or data is None or len(data) < 10:
            return {"status": "no_data", "records": 0}
        
        X, y_pressure, y_precip = self.prepare_data(data)
        
        results = {
            "status": "retrained",
            "records": len(data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if len(X) > 10:
            results["lstm"] = self.train_lstm(X, y_pressure)
            results["xgb"] = self.train_xgb(X, y_precip)
        
        return results

class AlertManager:
    def __init__(self, client):
        self.client = client
    
    def save_weather_data(self, location, lat, lon, weather_data):
        data = {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "temperature": weather_data['temperature'][0],
            "humidity": weather_data['humidity'][0],
            "pressure": weather_data['surface_pressure'][0],
            "precipitation": weather_data['precipitation'][0],
            "wind_speed": weather_data['windspeed'][0],
            "wind_gusts": weather_data['windgusts'][0],
            "cloud_cover": weather_data['cloudcover'][0],
            "dew_point": weather_data['dewpoint'][0]
        }
        response = self.client.table("weather_data").insert(data).execute()
        return response.data[0]['id'] if response.data else None
    
    def create_alerts(self, weather_data_id, alerts_data):
        alerts = []
        for alert in alerts_data:
            alerts.append({
                "weather_data_id": weather_data_id,
                "alert_type": alert['type'],
                "risk_level": alert['level'],
                "description": alert['description']
            })
        if alerts:
            self.client.table("alerts").insert(alerts).execute() 