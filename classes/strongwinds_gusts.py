import sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Importaciones opcionales para datos meteorológicos
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    METEO_AVAILABLE = True
except ImportError:
    METEO_AVAILABLE = False
    print("⚠️  Librerías meteorológicas no disponibles. Instalar con: pip install openmeteo-requests requests-cache retry-requests")

# Agregar el directorio padre al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

class StrongWindsPredictor:
    """Sistema de predicción de vientos fuertes y ráfagas"""
    
    def __init__(self, dataset_path=None):
        """Inicializar el predictor con el dataset"""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        
        # Usar dataset por defecto si no se especifica
        if dataset_path is None:
            self.dataset_path = self.project_root / 'public' / 'datasets' / 'unido.xlsx'
        else:
            self.dataset_path = Path(dataset_path)
        
        # Cargar dataset
        try:
            self.df = pd.read_excel(self.dataset_path)
            print(f"✅ Dataset cargado: {len(self.df)} registros")
        except FileNotFoundError:
            print(f"❌ No se encontró el dataset en: {self.dataset_path}")
            self.df = pd.DataFrame()
        
        # Configurar umbrales por ciudad
        self.setup_city_thresholds()
        
        # Calcular estadísticas del dataset si está disponible
        if not self.df.empty:
            self._calcular_estadisticas()

    def setup_city_thresholds(self):
        """Configurar umbrales específicos por ciudad"""
        self.umbral_viento = {
            # Región Andina
            "Bogotá": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Medellín": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Manizales": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Tunja": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Pasto": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Armenia": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Ibagué": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Popayán": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},

            # Región Caribe
            "Barranquilla": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},
            "Cartagena": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},
            "Santa Marta": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},
            "Riohacha": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},
            "San Andrés": {"fuerte": 14, "extremo": 22, "rafaga_fuerte": 20, "rafaga_extrema": 28},
            "Turbo": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},
            "Valledupar": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},
            "Montería": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},
            "Sincelejo": {"fuerte": 12, "extremo": 20, "rafaga_fuerte": 18, "rafaga_extrema": 25},

            # Llanos y Amazonía
            "Villavicencio": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Yopal": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Leticia": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Florencia": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Arauca": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},
            "Mocoa": {"fuerte": 8, "extremo": 15, "rafaga_fuerte": 12, "rafaga_extrema": 20},

            # Ciudades intermedias
            "Cali": {"fuerte": 10, "extremo": 18, "rafaga_fuerte": 15, "rafaga_extrema": 22},
            "Bucaramanga": {"fuerte": 10, "extremo": 18, "rafaga_fuerte": 15, "rafaga_extrema": 22},
            "Pereira": {"fuerte": 10, "extremo": 18, "rafaga_fuerte": 15, "rafaga_extrema": 22},
            "Neiva": {"fuerte": 10, "extremo": 18, "rafaga_fuerte": 15, "rafaga_extrema": 22},
            "Cúcuta": {"fuerte": 10, "extremo": 18, "rafaga_fuerte": 15, "rafaga_extrema": 22},
            "Buenaventura": {"fuerte": 10, "extremo": 18, "rafaga_fuerte": 15, "rafaga_extrema": 22},
            "Quibdó": {"fuerte": 10, "extremo": 18, "rafaga_fuerte": 15, "rafaga_extrema": 22},
            
            # Umbrales por defecto para ciudades no especificadas
            "default": {"fuerte": 10, "extremo": 20, "rafaga_fuerte": 15, "rafaga_extrema": 25}
        }

    def get_coordinates(self, location: str):
        """Obtiene las coordenadas de una ubicación usando Nominatim"""
        headers = {
            "User-Agent": "StrongWindsPredictor/1.0 (prediction@example.com)",
        }
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                results = response.json()
                if results:
                    return float(results[0]["lat"]), float(results[0]["lon"])
                else:
                    print(f"❌ Ubicación '{location}' no encontrada")
                    return None, None
            else:
                print(f"❌ Error al conectar con Nominatim API: {response.status_code}")
                return None, None
        except Exception as e:
            print(f"❌ Error obteniendo coordenadas: {e}")
            return None, None

    def _calcular_estadisticas(self):
        """Calcular estadísticas del dataset"""
        if self.df.empty:
            return
        
        # Verificar si existe la columna windgusts_10m, si no, usar windspeed_10m
        if 'windgusts_10m' not in self.df.columns:
            self.df['windgusts_10m'] = self.df['windspeed_10m'] * 1.5  # Estimación básica
        
        # Calcular predicciones de vientos para el dataset histórico
        df_con_vientos = self.analizar_vientos_y_rafagas(self.df)
        
        # Estadísticas generales
        total_registros = len(df_con_vientos)
        casos_viento_fuerte = len(df_con_vientos[df_con_vientos['alerta_viento'] == True])
        porcentaje_alertas = (casos_viento_fuerte / total_registros) * 100
        
        print(f"📊 Estadísticas del dataset:")
        print(f"   Total de registros: {total_registros:,}")
        print(f"   Casos de vientos fuertes: {casos_viento_fuerte:,}")
        print(f"   Porcentaje de alertas: {porcentaje_alertas:.2f}%")

    def analizar_vientos_y_rafagas(self, df, ciudad=None):
        """
        Analiza las condiciones de vientos fuertes y ráfagas en un DataFrame
        
        Args:
            df: DataFrame con columnas 'windspeed_10m', 'windgusts_10m'
            ciudad: Nombre de la ciudad para usar umbrales específicos
        
        Returns:
            DataFrame con columnas de análisis de vientos agregadas
        """
        # Copiar el DataFrame para no modificar el original
        df_resultado = df.copy()
        
        # Asegurar que existe la columna windgusts_10m
        if 'windgusts_10m' not in df_resultado.columns:
            df_resultado['windgusts_10m'] = df_resultado['windspeed_10m'] * 1.5
        
        # Obtener umbrales específicos para la ciudad
        if ciudad and ciudad in self.umbral_viento:
            umbrales = self.umbral_viento[ciudad]
        else:
            umbrales = self.umbral_viento["default"]
        
        # Aplicar criterios de vientos fuertes
        df_resultado['viento_fuerte'] = df_resultado['windspeed_10m'] >= umbrales['fuerte']
        df_resultado['viento_extremo'] = df_resultado['windspeed_10m'] >= umbrales['extremo']
        df_resultado['rafaga_fuerte'] = df_resultado['windgusts_10m'] >= umbrales['rafaga_fuerte']
        df_resultado['rafaga_extrema'] = df_resultado['windgusts_10m'] >= umbrales['rafaga_extrema']

        # Alerta general de viento peligroso
        df_resultado['alerta_viento'] = (
            df_resultado['viento_fuerte'] | 
            df_resultado['viento_extremo'] | 
            df_resultado['rafaga_fuerte'] | 
            df_resultado['rafaga_extrema']
        )
        
        # Nivel de riesgo general
        df_resultado['nivel_riesgo'] = 'sin_riesgo'
        df_resultado.loc[df_resultado['viento_fuerte'] | df_resultado['rafaga_fuerte'], 'nivel_riesgo'] = 'moderado'
        df_resultado.loc[df_resultado['viento_extremo'] | df_resultado['rafaga_extrema'], 'nivel_riesgo'] = 'alto'
        
        return df_resultado


    def get_weather_data(self, lat: float, lon: float, days: int = 7):
        """Obtiene datos meteorológicos usando Open-Meteo API"""
        if not METEO_AVAILABLE:
            print("❌ Librerías meteorológicas no disponibles")
            return None
        
        try:
            # Configurar cliente Open-Meteo con caché
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            # Parámetros para la API
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": [
                    "windspeed_10m",
                    "windgusts_10m",
                    "winddirection_10m",
                    "temperature_2m",
                    "surface_pressure"
                ],
                "timezone": "auto",
                "forecast_days": days
            }
            
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Procesar datos
            hourly = response.Hourly()
            hourly_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ),
                "windspeed_10m": hourly.Variables(0).ValuesAsNumpy(),
                "windgusts_10m": hourly.Variables(1).ValuesAsNumpy(),
                "winddirection_10m": hourly.Variables(2).ValuesAsNumpy(),
                "temperature_2m": hourly.Variables(3).ValuesAsNumpy(),
                "surface_pressure": hourly.Variables(4).ValuesAsNumpy(),
            }
            
            return pd.DataFrame(data=hourly_data)
            
        except Exception as e:
            print(f"❌ Error obteniendo datos meteorológicos: {e}")
            return None

    def predict_strong_winds(self, location: str, days: int = 3):
        """
        Predice el riesgo de vientos fuertes y ráfagas para una ubicación específica
        
        Args:
            location: Nombre de la ciudad o ubicación
            days: Número de días para la predicción (máximo 7)
        
        Returns:
            dict: Resultado de la predicción de vientos fuertes
        """
        print(f"🌬️ Analizando riesgo de vientos fuertes para: {location}")
        
        # Obtener coordenadas
        lat, lon = self.get_coordinates(location)
        if lat is None or lon is None:
            return {
                "error": f"No se pudieron obtener las coordenadas para '{location}'",
                "location": location,
                "coordinates": None
            }
        
        print(f"📍 Coordenadas: {lat:.4f}, {lon:.4f}")
        
        # Obtener datos meteorológicos
        weather_data = self.get_weather_data(lat, lon, min(days, 7))
        if weather_data is None:
            return {
                "error": "No se pudieron obtener los datos meteorológicos",
                "location": location,
                "coordinates": {"latitude": lat, "longitude": lon}
            }
        
        # Analizar condiciones de vientos
        df_analyzed = self.analizar_vientos_y_rafagas(weather_data, location)
        
        # Filtrar solo los próximos días solicitados
        total_hours = days * 24
        df_forecast = df_analyzed.head(total_hours)
        
        # Calcular estadísticas
        statistics = self._calculate_wind_statistics(df_forecast, location)
        
        # Generar resumen
        summary = self._generate_wind_summary(df_forecast, statistics, location)
        
        return {
            "location": location,
            "coordinates": {
                "latitude": float(lat),
                "longitude": float(lon)
            },
            "prediction_period_days": int(days),
            "city_thresholds": self.umbral_viento.get(location, self.umbral_viento["default"]),
            "summary": summary,
            "statistics": statistics,
            "hourly_forecast": self._format_hourly_wind_data(df_forecast),
            "raw_weather_data": {
                "windspeed_10m": [float(x) for x in weather_data["windspeed_10m"].tolist()],
                "windgusts_10m": [float(x) for x in weather_data["windgusts_10m"].tolist()],
                "winddirection_10m": [float(x) for x in weather_data["winddirection_10m"].tolist()],
                "temperature_2m": [float(x) for x in weather_data["temperature_2m"].tolist()],
                "surface_pressure": [float(x) for x in weather_data["surface_pressure"].tolist()],
                "time": [str(x) for x in weather_data["time"].tolist()]
            }
        }

    def _calculate_wind_statistics(self, df, ciudad):
        """Calcula estadísticas de riesgo de vientos fuertes"""
        total_hours = len(df)
        
        # Conteo por nivel de riesgo
        risk_counts = df['nivel_riesgo'].value_counts().to_dict()
        
        # Convertir a tipos nativos
        for key in risk_counts:
            risk_counts[key] = int(risk_counts[key])
        
        # Porcentajes
        risk_percentages = {}
        for risk, count in risk_counts.items():
            risk_percentages[risk] = float(count / total_hours * 100)
        
        # Conteos de alertas específicas
        alert_counts = {
            "viento_fuerte": int(df['viento_fuerte'].sum()),
            "viento_extremo": int(df['viento_extremo'].sum()),
            "rafaga_fuerte": int(df['rafaga_fuerte'].sum()),
            "rafaga_extrema": int(df['rafaga_extrema'].sum()),
            "alerta_general": int(df['alerta_viento'].sum())
        }
        
        # Estadísticas de viento
        wind_stats = {
            "max_windspeed": float(df['windspeed_10m'].max()),
            "mean_windspeed": float(df['windspeed_10m'].mean()),
            "max_windgusts": float(df['windgusts_10m'].max()),
            "mean_windgusts": float(df['windgusts_10m'].mean()),
            "predominant_direction": float(df['winddirection_10m'].mode().iloc[0]) if not df['winddirection_10m'].empty else 0.0
        }
        
        # Umbrales utilizados
        umbrales = self.umbral_viento.get(ciudad, self.umbral_viento["default"])
        
        return {
            "total_hours_analyzed": int(total_hours),
            "risk_level_counts": risk_counts,
            "risk_level_percentages": risk_percentages,
            "alert_counts": alert_counts,
            "wind_statistics": wind_stats,
            "thresholds_used": umbrales
        }

    def _generate_wind_summary(self, df, statistics, ciudad):
        """Genera un resumen de la predicción de vientos fuertes"""
        
        # Determinar nivel de riesgo general
        extreme_hours = statistics["alert_counts"]["viento_extremo"] + statistics["alert_counts"]["rafaga_extrema"]
        strong_hours = statistics["alert_counts"]["viento_fuerte"] + statistics["alert_counts"]["rafaga_fuerte"]
        
        if extreme_hours > 0:
            overall_risk = "MUY ALTO"
            risk_description = f"Condiciones de vientos extremos detectadas durante {extreme_hours} horas"
        elif strong_hours > 0:
            overall_risk = "ALTO"
            risk_description = f"Condiciones de vientos fuertes detectadas durante {strong_hours} horas"
        else:
            overall_risk = "BAJO"
            risk_description = "No se detectaron condiciones significativas de vientos fuertes"
        
        # Períodos críticos
        critical_periods = []
        current_period = None
        
        for idx, row in df.iterrows():
            if row['alerta_viento']:
                time_str = str(row['time'])
                if current_period is None:
                    current_period = {
                        "start_time": time_str,
                        "risk_level": row['nivel_riesgo'],
                        "max_windspeed": float(row['windspeed_10m']),
                        "max_windgusts": float(row['windgusts_10m']),
                        "wind_direction": float(row['winddirection_10m'])
                    }
                else:
                    current_period["end_time"] = time_str
                    current_period["max_windspeed"] = max(current_period["max_windspeed"], float(row['windspeed_10m']))
                    current_period["max_windgusts"] = max(current_period["max_windgusts"], float(row['windgusts_10m']))
            else:
                if current_period is not None:
                    if "end_time" not in current_period:
                        current_period["end_time"] = current_period["start_time"]
                    critical_periods.append(current_period)
                    current_period = None
        
        # Agregar el último período si existe
        if current_period is not None:
            if "end_time" not in current_period:
                current_period["end_time"] = current_period["start_time"]
            critical_periods.append(current_period)
        
        # Recomendaciones
        recommendations = []
        if extreme_hours > 0:
            recommendations.extend([
                "ALERTA MÁXIMA: Evitar actividades al aire libre",
                "Asegurar objetos sueltos y estructuras temporales",
                "Evitar circulación de vehículos altos o motocicletas",
                "Mantenerse alejado de árboles y estructuras elevadas"
            ])
        elif strong_hours > 0:
            recommendations.extend([
                "Precaución en actividades al aire libre",
                "Revisar amarres de objetos expuestos al viento",
                "Conducir con precaución, especialmente vehículos altos"
            ])
        else:
            recommendations.append("Condiciones de viento normales, sin precauciones especiales")
        
        return {
            "overall_risk_level": overall_risk,
            "risk_description": risk_description,
            "critical_periods": critical_periods,
            "recommendations": recommendations,
            "max_expected_windspeed": float(df['windspeed_10m'].max()),
            "max_expected_windgusts": float(df['windgusts_10m'].max()),
            "city_specific_thresholds": self.umbral_viento.get(ciudad, self.umbral_viento["default"])
        }

    def _format_hourly_wind_data(self, df, limit=48):
        """Formatea datos horarios de viento para la respuesta (limita a 48 horas)"""
        df_limited = df.head(limit)
        
        hourly_data = []
        for _, row in df_limited.iterrows():
            hourly_data.append({
                "time": str(row['time']),
                "windspeed": float(row['windspeed_10m']),
                "windgusts": float(row['windgusts_10m']),
                "wind_direction": float(row['winddirection_10m']),
                "temperature": float(row['temperature_2m']),
                "surface_pressure": float(row['surface_pressure']),
                "strong_wind_alert": bool(row['viento_fuerte']),
                "extreme_wind_alert": bool(row['viento_extremo']),
                "strong_gust_alert": bool(row['rafaga_fuerte']),
                "extreme_gust_alert": bool(row['rafaga_extrema']),
                "general_alert": bool(row['alerta_viento']),
                "risk_level": str(row['nivel_riesgo'])
            })
        
        return hourly_data


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del predictor
    predictor = StrongWindsPredictor()
    
    # Ejemplo de predicción para una ciudad
    print("🌬️ SISTEMA DE PREDICCIÓN DE VIENTOS FUERTES")
    print("=" * 60)
    
    # Probar con una ciudad
    city = "Barranquilla"
    result = predictor.predict_strong_winds(city, days=3)
    
    if "error" not in result:
        print(f"\n📊 Predicción para {city}:")
        print(f"   Nivel de riesgo: {result['summary']['overall_risk_level']}")
        print(f"   Descripción: {result['summary']['risk_description']}")
        print(f"   Períodos críticos: {len(result['summary']['critical_periods'])}")
        print(f"   Viento máximo esperado: {result['summary']['max_expected_windspeed']:.1f} m/s")
        print(f"   Ráfaga máxima esperada: {result['summary']['max_expected_windgusts']:.1f} m/s")
        
        if result['summary']['recommendations']:
            print("\n📋 Recomendaciones:")
            for rec in result['summary']['recommendations']:
                print(f"   • {rec}")
    else:
        print(f"❌ Error: {result['error']}")