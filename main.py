import os
import pickle
import xgboost  # Import xgboost
import requests
import numpy as np
import pandas as pd
import requests_cache
import tensorflow as tf  # Import tensorflow
import openmeteo_requests
from retry_requests import retry
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from classes.ValidationSystem import ValidationSystem

try:
    from supabase_trainer import SupabaseTrainer, AlertManager
    trainer = SupabaseTrainer()
    alert_manager = AlertManager(trainer.client) if trainer.client else None
    RETRAIN_ENABLED = True
    ALERTS_ENABLED = trainer.client is not None
except:
    RETRAIN_ENABLED = False
    ALERTS_ENABLED = False
    alert_manager = None

model_surface_pressure = tf.keras.models.load_model(
    "models/surface_pressure.keras") 
model_surface_pressure.summary() 
model_surface_pressure.summary() 

# Cargar el modelo pickle
try:
    with open("models/precipitation.pkl", 'rb') as file:
        model_precipitation = pickle.load(file)
except ModuleNotFoundError as e:
    if e.name == 'xgboost':
        raise HTTPException(
            status_code=500,
            detail="The model requires the 'xgboost' library. Please install it using 'pip install xgboost'."
        ) from e
    else:
        raise  # Re-raise the exception if it's not about xgboost
 
app = FastAPI()
origins = [
    "http://localhost:8000",  # Replace with the origin of your client
    "http://localhost",
    "http://127.0.0.1",
    "*",  
]

app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_coordinates(location: str): 
    headers = {
        "User-Agent": "PredictionModule/1.0 (once1234567890@example.com)",
    }
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
    response = requests.get(url, headers=headers)  # Added headers here
    if response.status_code == 200:
        results = response.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
        else:
            raise HTTPException(
                status_code=404, detail=f"Location '{location}' not found")
    else:
        raise HTTPException(
            status_code=500, detail="Failed to connect to Nominatim API")


def process_alerts(hourly_predictions, summary):
    alerts = []
    
    high_precip = summary.get("total_precipitation", 0) > 20
    strong_winds = summary.get("max_windspeed", 0) > 20
    frost_risk = summary.get("temperature_range", {}).get("min", 10) < 5
    high_risk_hours = summary.get("high_risk_hours", 0) > 8
    
    fog_hours = len([h for h in hourly_predictions if h["risks"]["fog_risk"] == "ALTO"])
    
    if high_precip:
        alerts.append({
            "type": "PRECIPITATION",
            "level": "ALTO",
            "description": f"Precipitación acumulada alta: {summary['total_precipitation']:.1f}mm"
        })
    
    if strong_winds:
        alerts.append({
            "type": "WIND",
            "level": "ALTO", 
            "description": f"Vientos fuertes: {summary['max_windspeed']:.1f} km/h"
        })
    
    if frost_risk:
        alerts.append({
            "type": "FROST",
            "level": "ALTO",
            "description": f"Riesgo de heladas: {summary['temperature_range']['min']:.1f}°C"
        })
    
    if fog_hours > 4:
        alerts.append({
            "type": "FOG",
            "level": "MEDIO" if fog_hours < 8 else "ALTO",
            "description": f"Niebla esperada durante {fog_hours} horas"
        })
    
    return alerts

@app.post("/predict")
def predict(location: str = Body(..., embed=True)):
    print("Location", location)
    lat, lon = get_coordinates(location)

    # Verificar si las coordenadas son válidas
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    data = get_weather_data(lat, lon)

    try:
        # Inicializar sistema de validación
        from classes.ValidationSystem import ValidationSystem
        validator = ValidationSystem()
        
        # Crear predicciones hora por hora para las próximas 24 horas
        hourly_predictions = []
        
        # Obtener datos horarios
        hourly_data = data.get('hourly_data', {})
        timestamps = hourly_data.get('timestamps', [])
        
        for i in range(len(timestamps)):
            # Obtener valores seguros
            temp_val = float(hourly_data.get('temperature', [])[i]) if i < len(hourly_data.get('temperature', [])) else 0.0
            precip_val = float(hourly_data.get('precipitation', [])[i]) if i < len(hourly_data.get('precipitation', [])) else 0.0
            humidity_val = float(hourly_data.get('humidity', [])[i]) if i < len(hourly_data.get('humidity', [])) else 0.0
            wind_val = float(hourly_data.get('windspeed', [])[i]) if i < len(hourly_data.get('windspeed', [])) else 0.0
            gust_val = float(hourly_data.get('windgusts', [])[i]) if i < len(hourly_data.get('windgusts', [])) else 0.0
            pressure_val = float(hourly_data.get('surface_pressure', [])[i]) if i < len(hourly_data.get('surface_pressure', [])) else 0.0
            cloud_val = float(hourly_data.get('cloudcover', [])[i]) if i < len(hourly_data.get('cloudcover', [])) else 0.0
            dew_val = float(hourly_data.get('dewpoint', [])[i]) if i < len(hourly_data.get('dewpoint', [])) else 0.0

            hour_data = {
                "hour": timestamps[i],
                "temperature": round(temp_val, 1),
                "precipitation": round(precip_val, 2),
                "humidity": round(humidity_val, 1),
                "windspeed": round(wind_val, 1),
                "windgusts": round(gust_val, 1),
                "surface_pressure": round(pressure_val, 1),
                "cloudcover": round(cloud_val, 0),
                "dewpoint": round(dew_val, 1)
            }
            
            # Calcular riesgos específicos para esta hora
            risks = {
                "precipitation_risk": "ALTO" if precip_val > 5 else "MEDIO" if precip_val > 1 else "BAJO",
                "wind_risk": "ALTO" if wind_val > 15 else "MEDIO" if wind_val > 8 else "BAJO",
                "fog_risk": "ALTO" if (temp_val - dew_val) < 2 and humidity_val > 95 else "BAJO",
                "frost_risk": "ALTO" if temp_val < 2 else "MEDIO" if temp_val < 5 else "BAJO"
            }
            
            hour_data["risks"] = risks
            hourly_predictions.append(hour_data)

        # Resumen general de 24 horas
        summary = {
            "temperature_range": {
                "min": min([h["temperature"] for h in hourly_predictions]) if hourly_predictions else 0,
                "max": max([h["temperature"] for h in hourly_predictions]) if hourly_predictions else 0
            },
            "total_precipitation": sum([h["precipitation"] for h in hourly_predictions]) if hourly_predictions else 0,
            "max_windspeed": max([h["windspeed"] for h in hourly_predictions]) if hourly_predictions else 0,
            "max_windgusts": max([h["windgusts"] for h in hourly_predictions]) if hourly_predictions else 0,
            "high_risk_hours": len([h for h in hourly_predictions if any(risk == "ALTO" for risk in h["risks"].values())]) if hourly_predictions else 0
        }

        # Generar reporte de validación del sistema
        validation_report = validator.generate_comprehensive_validation_report(hourly_data)
        
        # Alertas generales basadas en condiciones meteorológicas
        general_alerts = []
        total_precip = summary.get("total_precipitation", 0)
        max_wind = summary.get("max_windspeed", 0)
        temp_range = summary.get("temperature_range", {})
        high_risk = summary.get("high_risk_hours", 0)
        
        if isinstance(total_precip, (int, float)) and total_precip > 20:
            general_alerts.append("⚠️ Precipitación acumulada alta esperada")
        if isinstance(max_wind, (int, float)) and max_wind > 20:
            general_alerts.append("💨 Vientos fuertes esperados")
        if isinstance(temp_range, dict) and temp_range.get("min", 10) < 5:
            general_alerts.append("🥶 Riesgo de heladas")
        if isinstance(high_risk, (int, float)) and high_risk > 8:
            general_alerts.append("🚨 Múltiples horas de alto riesgo")

        alerts_data = process_alerts(hourly_predictions, summary)
        
        weather_data_id = None
        if ALERTS_ENABLED and alert_manager and alerts_data:
            try:
                weather_data_id = alert_manager.save_weather_data(
                    location, lat, lon, data['hourly_data']
                )
                if weather_data_id:
                    alert_manager.create_alerts(weather_data_id, alerts_data)
            except Exception as e:
                print(f"Error guardando alertas: {e}")

        return {
            "location": location,
            "coordinates": {"lat": lat, "lon": lon},
            "forecast_period": "Próximas 24 horas",
            "generated_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "hourly_predictions": hourly_predictions,
            "summary_24h": summary,
            "general_alerts": general_alerts if general_alerts else ["✅ Sin alertas meteorológicas significativas"],
            "validation_report": {
                "overall_system_validity": f"{validation_report.get('overall_system_score', 0)}%",
                "data_quality": f"{validation_report.get('data_sources_evaluation', {}).get('openmeteo_api', {}).get('overall_validity', 0)}%",
                "models_status": {
                    "surface_pressure": f"{validation_report.get('models_evaluation', {}).get('surface_pressure_lstm', {}).get('overall_validity', 0)}%",
                    "precipitation": f"{validation_report.get('models_evaluation', {}).get('precipitation_xgboost', {}).get('overall_validity', 0)}%"
                },
                "prediction_classes_avg": f"{np.mean([eval.get('overall_validity', 0) for eval in validation_report.get('prediction_classes_evaluation', {}).values()]):.1f}%",
                "recommendations": validation_report.get('recommendations', []),
                "last_validation": validation_report.get('validation_timestamp', 'N/A')
            },
            "alerts": alerts_data,
            "alerts_saved": len(alerts_data) if weather_data_id else 0,
            "database_id": weather_data_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/validation")
def get_validation_report():
    """Endpoint para obtener un reporte detallado de validación del sistema"""
    try:
        from classes.ValidationSystem import ValidationSystem
        validator = ValidationSystem()
        
        # Obtener datos de muestra para evaluar calidad de API
        sample_data = get_weather_data(4.6534109, -74.0836547)  # Coordenadas de Bogotá
        hourly_data = sample_data.get('hourly_data', {})
        
        # Generar reporte completo
        full_report = validator.generate_comprehensive_validation_report(hourly_data)
        
        return {
            "status": "success",
            "validation_report": full_report,
            "summary": {
                "overall_system_score": f"{full_report.get('overall_system_score', 0)}%",
                "total_components_evaluated": (
                    len(full_report.get('models_evaluation', {})) + 
                    len(full_report.get('data_sources_evaluation', {})) + 
                    len(full_report.get('prediction_classes_evaluation', {}))
                ),
                "system_status": (
                    "EXCELENTE" if full_report.get('overall_system_score', 0) > 85 else
                    "BUENO" if full_report.get('overall_system_score', 0) > 70 else
                    "REQUIERE_ATENCION"
                )
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte de validación: {str(e)}")

def get_weather_data(lat: float, lon: float):
    session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "surface_pressure", 
            "precipitation",
            "relative_humidity_2m",
            "cloudcover",
            "windspeed_10m",
            "windgusts_10m",
            "dewpoint_2m"
        ],
        "forecast_days": 1,  # Solo próximas 24 horas
        "timezone": "America/Bogota"
    }

    try:
        responses = client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()

        # Obtener todas las variables para las próximas 24 horas
        temperature = hourly.Variables(0).ValuesAsNumpy()
        surface_pressure = hourly.Variables(1).ValuesAsNumpy()
        precipitation = hourly.Variables(2).ValuesAsNumpy()
        humidity = hourly.Variables(3).ValuesAsNumpy()
        cloudcover = hourly.Variables(4).ValuesAsNumpy()
        windspeed = hourly.Variables(5).ValuesAsNumpy()
        windgusts = hourly.Variables(6).ValuesAsNumpy()
        dewpoint = hourly.Variables(7).ValuesAsNumpy()

        # Crear timestamps para las próximas 24 horas
        current_time = pd.Timestamp.now(tz='America/Bogota')
        timestamps = [current_time + pd.Timedelta(hours=i) for i in range(24)]
        
        # Truncar a exactamente 24 horas si hay más datos
        if len(temperature) > 24:
            temperature = temperature[:24]
            surface_pressure = surface_pressure[:24]
            precipitation = precipitation[:24]
            humidity = humidity[:24]
            cloudcover = cloudcover[:24]
            windspeed = windspeed[:24]
            windgusts = windgusts[:24]
            dewpoint = dewpoint[:24]

        return {
            'hourly_data': {
                'timestamps': [ts.strftime('%Y-%m-%d %H:00') for ts in timestamps],
                'temperature': temperature.tolist(),
                'surface_pressure': surface_pressure.tolist(),
                'precipitation': precipitation.tolist(),
                'humidity': humidity.tolist(),
                'cloudcover': cloudcover.tolist(),
                'windspeed': windspeed.tolist(),
                'windgusts': windgusts.tolist(),
                'dewpoint': dewpoint.tolist()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def make_prediction(model, data, model_type="keras"):
    try:
        if model_type == "keras":
            lstm_input = data['lstm_input']
            print(f"Original LSTM input shape: {lstm_input.shape}")
            
            batch_size = 32
            num_samples = lstm_input.shape[0]
            
            # Ensure we have the correct input shape (batch_size, 24, 12)
            if num_samples < batch_size:
                padding = np.zeros((batch_size - num_samples, 24, 12), dtype=np.float32)
                lstm_input = np.concatenate([lstm_input, padding], axis=0)
            else:
                lstm_input = lstm_input[:batch_size]
            
            print(f"Final LSTM input shape: {lstm_input.shape}")
            
            # Make prediction
            prediction = model.predict(lstm_input)
            
            # If we padded the input, remove the padding from the prediction
            if num_samples < batch_size:
                prediction = prediction[:num_samples]
            
            return prediction
            
        elif model_type == "pickle":
            return model.predict(data['xgb_input'])
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction with {model_type} model: {str(e)}"
        )


@app.get("/alerts/{location}")
def get_alerts(location: str):
    if not ALERTS_ENABLED:
        raise HTTPException(status_code=503, detail="Alertas no disponibles")
    
    response = trainer.client.table("alerts")\
        .select("*, weather_data!inner(location)")\
        .eq("weather_data.location", location)\
        .order("created_at", desc=True)\
        .limit(50)\
        .execute()
    
    return {"alerts": response.data}

@app.get("/alerts/recent/{hours}")
def get_recent_alerts(hours: int = 24):
    if not ALERTS_ENABLED:
        raise HTTPException(status_code=503, detail="Alertas no disponibles")
    
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    response = trainer.client.table("alerts")\
        .select("*, weather_data(location, latitude, longitude)")\
        .gte("created_at", cutoff.isoformat())\
        .order("created_at", desc=True)\
        .execute()
    
    return {"alerts": response.data, "period_hours": hours}

@app.post("/retrain")
def retrain():
    if not RETRAIN_ENABLED:
        raise HTTPException(status_code=503, detail="Reentrenamiento no disponible")
    return trainer.retrain()

@app.get("/retrain/status")
def retrain_status():
    if not RETRAIN_ENABLED:
        return {"enabled": False}
    return {"enabled": True, "connected": trainer.client is not None}

# Iniciar la aplicación FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=int(os.getenv('PORT', '8000')))
