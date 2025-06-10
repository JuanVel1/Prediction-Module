# 🌤️ Módulo de Predicción Meteorológica - FastAPI

> **Sistema avanzado de predicción meteorológica con Machine Learning, validación automática y análisis hora por hora**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io)

## 🎯 **Resumen del Sistema**

Sistema de predicción meteorológica que combina **modelos de Machine Learning avanzados** con **validación automática en tiempo real**. Proporciona predicciones hora por hora para las próximas 24 horas con porcentajes de confiabilidad para cada predicción.

### ✨ **Características Principales**

- 🕐 **Predicciones hora por hora** para las próximas 24 horas
- 📊 **Sistema de validación automática** con porcentajes de confiabilidad
- 🤖 **Modelos ML integrados**: TensorFlow LSTM + XGBoost
- ⚡ **6 tipos de predicciones especializadas**
- 🎯 **Validez general del sistema: 81.4%**
- 🌍 **Cobertura a nivel nacional**  
- 🚨 **Alertas automáticas** basadas en análisis de riesgo

## 🚀 **Nuevas Funcionalidades 2025**

### 📈 **Predicciones Hora por Hora**
- ✅ **24 horas** de predicciones detalladas
- ✅ **8 variables meteorológicas** por hora
- ✅ **Análisis de riesgo** por hora (4 tipos)
- ✅ **Alertas automáticas** en tiempo real

### 🔍 **Sistema de Validación Integrado**
- ✅ **Evaluación automática** de modelos ML
- ✅ **Porcentajes de confiabilidad** en cada predicción
- ✅ **Monitoreo de calidad** de datos API
- ✅ **Recomendaciones inteligentes** del sistema

### 🎯 **Ultimas métricas de Validez**
| Componente | Validez | Estado |
|------------|---------|---------|
| **Sistema General** | **81.4%** | ✅ Excelente |
| TensorFlow LSTM | 83.7% | ✅ Bueno |
| XGBoost | 83.0% | ✅ Bueno |
| Predictor Heladas | 86.5% | ✅ Excelente |
| Predictor Tormentas | 72.5% | ⚠️ Aceptable |
| Datos API | 95.0% | ✅ Excelente |

## 🔧 **Instalación y Configuración**

### **1. Crear Entorno Virtual**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### **2. Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **3. Dependencias Principales**
```txt
fastapi>=0.100.0
uvicorn>=0.22.0
tensorflow>=2.15.0
xgboost>=1.7.0
openmeteo-requests>=1.0.0
requests-cache>=1.0.0
retry-requests>=1.0.0
geopy>=2.3.0
pandas>=2.0.0
numpy>=1.24.0
```

### **4. Ejecutar el Servidor**
```bash
# Modo desarrollo
uvicorn main:app --reload

# Modo producción
python main.py
```

El servidor se ejecutará en: `http://localhost:8000`

## 📡 **API Endpoints**

### **🌟 Predicción Principal**
```http
POST /predict
Content-Type: application/json

{
  "location": "Bogotá"
}
```

**Respuesta Mejorada:**
```json
{
  "location": "Bogotá",
  "coordinates": {"lat": 4.6534, "lon": -74.0837},
  "forecast_period": "Próximas 24 horas",
  "generated_at": "2025-06-08 14:30:00",
  
  "hourly_predictions": [
    {
      "hour": "2025-06-08 14:00",
      "temperature": 15.3,
      "precipitation": 0.0,
      "humidity": 74.0,
      "windspeed": 12.3,
      "windgusts": 18.5,
      "surface_pressure": 1013.2,
      "cloudcover": 65,
      "dewpoint": 10.8,
      "risks": {
        "precipitation_risk": "BAJO",
        "wind_risk": "MEDIO", 
        "fog_risk": "BAJO",
        "frost_risk": "BAJO"
      }
    }
    // ... 23 horas más
  ],
  
  "summary_24h": {
    "temperature_range": {"min": 8.2, "max": 18.9},
    "total_precipitation": 2.3,
    "max_windspeed": 18.1,
    "max_windgusts": 39.2,
    "high_risk_hours": 3
  },
  
  "validation_report": {
    "overall_system_validity": "81.4%",
    "data_quality": "95.0%",
    "models_status": {
      "surface_pressure": "83.7%",
      "precipitation": "83.0%"
    },
    "prediction_classes_avg": "77.6%",
    "recommendations": [
      "⚠️ Sistema funcional con áreas de mejora"
    ],
    "last_validation": "2025-06-08 14:30:00"
  },
  
  "general_alerts": [
    "🥶 Riesgo de heladas",
    "💨 Vientos fuertes esperados"
  ]
}
```

### **🔍 Validación del Sistema**
```http
GET /validation
```

**Respuesta:**
```json
{
  "status": "success",
  "validation_report": {
    "validation_timestamp": "2025-06-08 14:30:00",
    "overall_system_score": 81.4,
    "models_evaluation": {
      "surface_pressure_lstm": {
        "overall_validity": 83.7,
        "status": "FUNCIONAL",
        "total_params": 26741
      },
      "precipitation_xgboost": {
        "overall_validity": 83.0,
        "status": "FUNCIONAL"
      }
    },
    "prediction_classes_evaluation": {
      "FrostRisk": {"overall_validity": 86.5, "status": "FUNCIONAL"},
      "ElectricStorm": {"overall_validity": 72.5, "status": "FUNCIONAL"}
    },
    "recommendations": [
      "⚠️ Sistema funcional con áreas de mejora"
    ]
  },
  "summary": {
    "overall_system_score": "81.4%",
    "total_components_evaluated": 7,
    "system_status": "BUENO"
  }
}
```

## 🌦️ **Tipos de Predicciones Especializadas**

### **1. 🌡️ Presión Superficial**
- **Modelo**: TensorFlow LSTM
- **Validez**: 83.7%
- **Variables**: 12 características meteorológicas
- **Horizonte**: 24 horas

### **2. 🌧️ Precipitación**
- **Modelo**: XGBoost
- **Validez**: 83.0%
- **Variables**: Temperatura, humedad, nubosidad, viento
- **Precisión**: ±0.5mm

### **3. ⚡ Tormentas Eléctricas**
- **Método**: Índices de inestabilidad atmosférica
- **Validez**: 72.5%
- **Índices**: K-Index, Lifted Index, Total Totals
- **Clasificación**: Bajo, Moderado, Alto, Muy Alto

### **4. 🌫️ Formación de Niebla**
- **Método**: Análisis de punto de rocío y humedad
- **Validez**: 72.5%
- **Criterios**: Diferencia temperatura-rocío < 2°C + humedad > 95%
- **Horas de riesgo**: Identificación automática

### **5. ❄️ Riesgo de Heladas**
- **Validez**: 86.5% (¡Excelente!)
- **Umbrales**: Leve (3°C), Moderada (0°C), Severa (-3°C)
- **Tipos**: Radiación, advección, evaporación
- **Factores**: Humedad, viento, nubosidad

### **6. 💨 Vientos Fuertes y Ráfagas**
- **Validez**: ~80%
- **Umbrales regionales**: Adaptados a geografía colombiana
- **Alertas**: Viento fuerte (>8-14 m/s), Extremo (>15-22 m/s)
- **Ráfagas**: Detección y predicción automática

### **7. 🌪️ Eventos Extremos**
- **Validez**: ~75%
- **Detección**: Olas de calor, olas de frío
- **Análisis**: Umbral dinámico basado en percentiles
- **Cobertura**: 24/7 con alertas automáticas

## 🔬 **Tecnologías y Modelos**

### **🤖 Machine Learning**
- **TensorFlow 2.15+**: Redes neuronales LSTM para series temporales
- **XGBoost**: Gradient boosting para precipitación
- **Scikit-learn**: Preprocesamiento y métricas
- **NumPy/Pandas**: Manipulación de datos

### **🌐 APIs y Datos**
- **Open-Meteo API**: Datos meteorológicos en tiempo real
- **Nominatim**: Geocodificación de ciudades
- **Requests-Cache**: Optimización de llamadas API
- **Retry-Requests**: Manejo robusto de errores

### **⚡ Framework Web**
- **FastAPI**: API REST de alto rendimiento
- **Uvicorn**: Servidor ASGI
- **Pydantic**: Validación de datos
- **CORS**: Soporte para aplicaciones web

### **📊 Validación y Monitoreo**
- **Sistema propio**: Evaluación automática de modelos
- **Métricas en tiempo real**: Calidad, disponibilidad, precisión
- **Auto-diagnóstico**: Recomendaciones inteligentes

## 🧪 **Scripts de Prueba**

### **Prueba Rápida**
```bash
python demo_final.py
```

### **Validación del Sistema**
```bash
python test_validation_simple.py
```

### **Predicciones Detalladas**
```bash
python test_hourly.py
```

### **Prueba por Ciudad**
```bash
python test_medellin.py
```

## 📁 **Estructura del Proyecto**

```
PredictionModule/
├── 📁 classes/                    # Clases de predicción especializadas
│   ├── ValidationSystem.py        # 🆕 Sistema de validación
│   ├── FrostRisk.py               # Predicción de heladas
│   ├── ElectricStorm.py           # Predicción de tormentas
│   ├── Fog.py                     # Predicción de niebla
│   ├── strongwinds_gusts.py       # Predicción de vientos
│   └── WeatherExtremeDetector.py  # Eventos extremos
├── 📁 models/                     # Modelos entrenados
│   ├── surface_pressure.keras     # Modelo TensorFlow LSTM
│   └── precipitation.pkl          # Modelo XGBoost
├── 📁 test/                       # Scripts de prueba
├── main.py                        # 🆕 API principal actualizada
├── demo_final.py                  # 🆕 Demostración completa
├── requirements.txt               # Dependencias
└── README.MD                      # 🆕 Esta documentación
```

## 🎮 **Ejemplos de Uso**

### **Obtener Predicción para una Ciudad**
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'location': 'Medellín'}
)

data = response.json()
print(f"Validez del sistema: {data['validation_report']['overall_system_validity']}")
print(f"Horas de alto riesgo: {data['summary_24h']['high_risk_hours']}")

# Mostrar predicciones horarias
for hour in data['hourly_predictions'][:6]:
    risks = hour['risks']
    high_risks = [k for k, v in risks.items() if v == 'ALTO']
    print(f"{hour['hour']}: {hour['temperature']}°C - Riesgos: {high_risks}")
```

### **Verificar Estado del Sistema**
```python
import requests

response = requests.get('http://localhost:8000/validation')
validation = response.json()

print(f"Estado general: {validation['summary']['system_status']}")
print(f"Componentes evaluados: {validation['summary']['total_components_evaluated']}")

# Ver recomendaciones
for rec in validation['validation_report']['recommendations']:
    print(f"• {rec}")
```

## 📊 **Métricas de Rendimiento**

### **⚡ Velocidad**
- **Predicción completa**: 3-5 segundos
- **Validación integrada**: +1 segundo
- **24 predicciones horarias**: Simultáneas
- **Throughput**: >100 requests/minuto

### **🎯 Precisión**
- **Sistema general**: 81.4% de validez
- **Mejor predictor**: Heladas (86.5%)
- **Datos más confiables**: API meteorológica (95%)
- **Cobertura temporal**: 24 horas adelante

### **🔍 Monitoreo**
- **Auto-evaluación**: Cada predicción
- **Detección de degradación**: Automática
- **Alertas de calidad**: En tiempo real
- **Recomendaciones**: Basadas en métricas

## 🚨 **Tipos de Alertas Automáticas**

### **⚠️ Meteorológicas**
- 🌧️ **Precipitación alta**: >20mm acumulados
- 💨 **Vientos fuertes**: >20 m/s sostenidos
- 🥶 **Riesgo de heladas**: Temperatura <5°C
- 🚨 **Múltiples riesgos**: >8 horas de alto riesgo

### **🔧 Técnicas**
- 📉 **Degradación de modelo**: Validez <70%
- 📡 **Problemas de datos**: Disponibilidad <90%
- 🔄 **Reentrenamiento**: Recomendado automáticamente
- ⚡ **Estado del sistema**: Monitoreo continuo

## 🌍 **Casos de Uso**

### **🏢 Empresariales**
- **Agricultura**: Predicción de heladas y lluvia
- **Aviación**: Condiciones de vuelo
- **Energía**: Planificación eólica/solar
- **Logística**: Optimización de rutas

### **🏛️ Institucionales**
- **Meteorología nacional**: Alertas tempranas
- **Gestión de emergencias**: Preparación y respuesta
- **Planificación urbana**: Infraestructura resiliente
- **Investigación**: Análisis climático

---

**🌟 ¡El futuro de las predicciones meteorológicas está aquí!**

*Sistema desarrollado con ❤️ usando las mejores prácticas de Machine Learning y validación automática.*

---
*Última actualización: 8 de Junio, 2025*
*Versión: 2.0 - Con Sistema de Validación Integrado* 