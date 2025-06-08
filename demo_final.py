import requests
import time
import subprocess
import threading
from classes.ValidationSystem import ValidationSystem

def start_server():
    """Función para iniciar el servidor en un hilo separado"""
    try:
        subprocess.run(["python", "main.py"], check=True)
    except:
        pass

def test_server_connection(max_attempts=10):
    """Verifica si el servidor está disponible"""
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://localhost:8000/docs', timeout=2)
            if response.status_code == 200:
                return True
        except:
            time.sleep(1)
            print(f"⏳ Esperando servidor... intento {attempt + 1}/{max_attempts}")
    return False

def demo_predictions_with_validation():
    """Demonstración de predicciones con validación integrada"""
    
    print("🚀 === DEMOSTRACIÓN FINAL: PREDICCIONES CON VALIDACIÓN ===\n")
    
    # Primero hacer validación offline
    print("🔍 1. VALIDACIÓN OFFLINE DEL SISTEMA")
    print("-" * 50)
    
    try:
        validator = ValidationSystem()
        
        # Evaluar modelos principales
        tf_eval = validator.evaluate_tensorflow_model("models/surface_pressure.keras", "LSTM")
        xgb_eval = validator.evaluate_xgboost_model("models/precipitation.pkl")
        
        print(f"🤖 Modelo TensorFlow (Presión): {tf_eval.get('overall_validity', 0)}%")
        print(f"🌧️ Modelo XGBoost (Precipitación): {xgb_eval.get('overall_validity', 0)}%")
        
        # Evaluar algunas clases clave
        frost_eval = validator.evaluate_prediction_class("FrostRisk")
        storm_eval = validator.evaluate_prediction_class("ElectricStorm")
        
        print(f"❄️ Predictor de Heladas: {frost_eval.get('overall_validity', 0)}%")
        print(f"⚡ Predictor de Tormentas: {storm_eval.get('overall_validity', 0)}%")
        
        # Calcular promedio
        avg_validity = (tf_eval.get('overall_validity', 0) + 
                       xgb_eval.get('overall_validity', 0) + 
                       frost_eval.get('overall_validity', 0) + 
                       storm_eval.get('overall_validity', 0)) / 4
        
        print(f"\n✅ Validez promedio del sistema: {avg_validity:.1f}%")
        print()
        
    except Exception as e:
        print(f"❌ Error en validación offline: {e}")
        print()
    
    # Intentar usar el servidor si está disponible
    print("🌐 2. PREDICCIONES EN LÍNEA (Requiere servidor)")
    print("-" * 50)
    
    try:
        # Probar conexión simple
        response = requests.post('http://localhost:8000/predict', 
                               json={'location': 'Bogotá'}, 
                               timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"📍 Ubicación: {data.get('location', 'N/A')}")
            print(f"⏰ Período: {data.get('forecast_period', 'N/A')}")
            
            # Mostrar validación integrada
            validation = data.get('validation_report', {})
            print(f"\n🔍 VALIDACIÓN INTEGRADA:")
            print(f"  🎯 Validez general: {validation.get('overall_system_validity', 'N/A')}")
            print(f"  📊 Calidad datos API: {validation.get('data_quality', 'N/A')}")
            
            models_status = validation.get('models_status', {})
            print(f"  🤖 Estado modelos:")
            for model, validity in models_status.items():
                print(f"    • {model}: {validity}")
            
            # Mostrar resumen meteorológico
            summary = data.get('summary_24h', {})
            print(f"\n🌤️ RESUMEN METEOROLÓGICO:")
            temp_range = summary.get('temperature_range', {})
            print(f"  🌡️ Temperatura: {temp_range.get('min', 0)}°C - {temp_range.get('max', 0)}°C")
            print(f"  💨 Viento máximo: {summary.get('max_windspeed', 0):.1f} m/s")
            print(f"  ⚠️ Horas de riesgo: {summary.get('high_risk_hours', 0)}")
            
            # Mostrar algunas predicciones
            hourly = data.get('hourly_predictions', [])
            print(f"\n⏰ PREDICCIONES HORARIAS (Muestra de {min(4, len(hourly))}):")
            for i, hour in enumerate(hourly[:4]):
                risks = hour.get('risks', {})
                high_risks = [k.replace('_risk', '').upper() for k, v in risks.items() if v == 'ALTO']
                risk_indicator = "🔴" if high_risks else "🟢"
                
                print(f"  {risk_indicator} {hour.get('hour', '')}: {hour.get('temperature', 0)}°C, "
                      f"💧{hour.get('humidity', 0)}%, ⛅{hour.get('cloudcover', 0)}%")
            
            print(f"\n📈 Total: {len(hourly)} predicciones horarias disponibles")
            
            # Mostrar recomendaciones
            recommendations = validation.get('recommendations', [])
            if recommendations:
                print(f"\n💡 RECOMENDACIONES DEL SISTEMA:")
                for rec in recommendations[:3]:  # Mostrar máximo 3
                    print(f"  • {rec}")
            
            print(f"\n✅ Predicciones con validación completadas exitosamente!")
            
        else:
            print(f"❌ Error del servidor: {response.status_code}")
            
    except requests.exceptions.RequestException:
        print("🔌 Servidor no disponible - Ejecutando demo offline")
        print("💡 Para usar predicciones en línea, ejecuta: python main.py")
        
        # Demo offline con datos simulados
        print(f"\n📊 DEMO OFFLINE - ESTRUCTURA DE RESPUESTA:")
        print(f"  🎯 Validez general del sistema: {avg_validity:.1f}%")
        print(f"  📍 Ubicación: Bogotá")
        print(f"  ⏰ Período: Próximas 24 horas")
        print(f"  📈 Predicciones horarias: 24 horas")
        print(f"  🔍 Validación: Incluida en cada respuesta")
        print(f"  💡 Recomendaciones: Automáticas basadas en calidad")
    
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
    
    print(f"\n🎉 === DEMOSTRACIÓN COMPLETADA ===")
    print(f"📋 Funcionalidades implementadas:")
    print(f"  ✅ Predicciones hora por hora (24h)")
    print(f"  ✅ Validación automática de modelos")
    print(f"  ✅ Porcentajes de confiabilidad")
    print(f"  ✅ Evaluación de calidad de datos")
    print(f"  ✅ Recomendaciones del sistema")
    print(f"  ✅ Análisis de riesgo por hora")
    print(f"  ✅ Alertas meteorológicas automáticas")

if __name__ == "__main__":
    demo_predictions_with_validation() 