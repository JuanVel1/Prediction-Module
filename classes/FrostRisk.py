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

class FrostRiskPredictor:
    """Sistema de predicción de riesgo de heladas"""
    
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
        
        # Umbrales de temperatura para heladas (°C)
        self.umbrales = {
            'leve': 3,          # Riesgo de helada leve: temperatura entre 0°C y 3°C
            'moderada': 0,      # Riesgo de helada moderada: temperatura entre -3°C y 0°C
            'severa': -3        # Riesgo de helada severa: temperatura menor a -3°C
        }

        # Condiciones favorables para heladas
        self.condiciones_favorables = {
            'humedad_max': 70,  # Humedad relativa máxima para heladas de radiación
            'viento_max': 8,    # Velocidad máxima del viento (m/s) para heladas de radiación
            'nubosidad_max': 25 # Cobertura de nubes máxima (%) para heladas de radiación
        }
        
        # Calcular estadísticas del dataset si está disponible
        if not self.df.empty:
            self._calcular_estadisticas()

    def get_coordinates(self, location: str):
        """Obtiene las coordenadas de una ubicación usando Nominatim"""
        headers = {
            "User-Agent": "FrostRiskPredictor/1.0 (prediction@example.com)",
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
        
        # Calcular predicciones de heladas para el dataset histórico
        df_con_heladas = self.analizar_condiciones_helada(self.df)
        
        # Estadísticas generales
        total_registros = len(df_con_heladas)
        casos_helada = len(df_con_heladas[df_con_heladas['riesgo_helada'] != 'sin_riesgo'])
        porcentaje_heladas = (casos_helada / total_registros) * 100
        
        print(f"📊 Estadísticas del dataset:")
        print(f"   Total de registros: {total_registros:,}")
        print(f"   Casos de heladas: {casos_helada:,}")
        print(f"   Porcentaje de heladas: {porcentaje_heladas:.2f}%")

    def calcular_punto_rocio(self, temp, humedad):
        """
        Calcula el punto de rocío usando la fórmula de Magnus-Tetens
        """
        a = 17.27
        b = 237.7

        # Calcular la temperatura del punto de rocío
        alpha = ((a * temp) / (b + temp)) + np.log(humedad/100.0)
        return (b * alpha) / (a - alpha)

    def calcular_riesgo_helada(self, temp, humedad, viento, nubosidad):
        """
        Calcula el riesgo de helada basado en las condiciones meteorológicas
        """
        # Calcular punto de rocío
        punto_rocio = self.calcular_punto_rocio(temp, humedad)

        # Inicializar variables de riesgo
        riesgo = 'sin_riesgo'
        tipo_helada = []
        factores_riesgo = []

        # Evaluar temperatura
        if temp <= self.umbrales['severa']:
            riesgo = 'severa'
        elif temp <= self.umbrales['moderada']:
            riesgo = 'moderada'
        elif temp <= self.umbrales['leve']:
            riesgo = 'leve'

        # Determinar tipo de helada y factores de riesgo
        if riesgo != 'sin_riesgo':
            # Helada de radiación
            if (humedad <= self.condiciones_favorables['humedad_max'] and
                viento <= self.condiciones_favorables['viento_max'] and
                nubosidad <= self.condiciones_favorables['nubosidad_max']):
                tipo_helada.append('radiación')
                factores_riesgo.extend([
                    'cielo_despejado' if nubosidad <= 25 else None,
                    'viento_calmo' if viento <= 2 else None,
                    'baja_humedad' if humedad <= 70 else None
                ])

            # Helada de advección
            if viento > self.condiciones_favorables['viento_max']:
                tipo_helada.append('advección')
                factores_riesgo.append('viento_fuerte')

            # Helada de evaporación
            if punto_rocio < 0 and humedad < 50:
                tipo_helada.append('evaporación')
                factores_riesgo.append('aire_seco')

        # Limpiar None de factores_riesgo
        factores_riesgo = [f for f in factores_riesgo if f is not None]

        return {
            'riesgo': riesgo,
            'tipo_helada': tipo_helada,
            'factores_riesgo': factores_riesgo,
            'punto_rocio': punto_rocio
        }

    def analizar_condiciones_helada(self, df):
        """
        Analiza las condiciones de helada en un DataFrame

        Args:
            df: DataFrame con columnas 'temperature_2m', 'relativehumidity_2m',
                'windspeed_10m', 'cloudcover'
        """
        # Copiar el DataFrame para no modificar el original
        df_resultado = df.copy()

        # Aplicar el análisis de riesgo de helada
        resultados = df.apply(lambda row: self.calcular_riesgo_helada(
            row['temperature_2m'],
            row['relativehumidity_2m'],
            row['windspeed_10m'],
            row['cloudcover']
        ), axis=1)

        # Extraer resultados
        df_resultado['riesgo_helada'] = resultados.apply(lambda x: x['riesgo'])
        df_resultado['tipo_helada'] = resultados.apply(lambda x: x['tipo_helada'])
        df_resultado['factores_riesgo'] = resultados.apply(lambda x: x['factores_riesgo'])
        df_resultado['punto_rocio'] = resultados.apply(lambda x: x['punto_rocio'])

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
                    "temperature_2m",
                    "relativehumidity_2m", 
                    "windspeed_10m",
                    "cloudcover"
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
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                "relativehumidity_2m": hourly.Variables(1).ValuesAsNumpy(),
                "windspeed_10m": hourly.Variables(2).ValuesAsNumpy(),
                "cloudcover": hourly.Variables(3).ValuesAsNumpy(),
            }
            
            return pd.DataFrame(data=hourly_data)
            
        except Exception as e:
            print(f"❌ Error obteniendo datos meteorológicos: {e}")
            return None

    def predict_frost_risk(self, location: str, days: int = 3):
        """
        Predice el riesgo de heladas para una ubicación específica
        
        Args:
            location: Nombre de la ciudad o ubicación
            days: Número de días para la predicción (máximo 7)
        
        Returns:
            dict: Resultado de la predicción de heladas
        """
        print(f"🔍 Analizando riesgo de heladas para: {location}")
        
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
        
        # Analizar condiciones de helada
        df_analyzed = self.analizar_condiciones_helada(weather_data)
        
        # Filtrar solo los próximos días solicitados
        total_hours = days * 24
        df_forecast = df_analyzed.head(total_hours)
        
        # Calcular estadísticas
        statistics = self._calculate_frost_statistics(df_forecast)
        
        # Generar resumen
        summary = self._generate_frost_summary(df_forecast, statistics)
        
        return {
            "location": location,
            "coordinates": {
                "latitude": float(lat),
                "longitude": float(lon)
            },
            "prediction_period_days": int(days),
            "summary": summary,
            "statistics": statistics,
            "hourly_forecast": self._format_hourly_data(df_forecast),
            "raw_weather_data": {
                "temperature_2m": [float(x) for x in weather_data["temperature_2m"].tolist()],
                "relativehumidity_2m": [float(x) for x in weather_data["relativehumidity_2m"].tolist()],
                "windspeed_10m": [float(x) for x in weather_data["windspeed_10m"].tolist()],
                "cloudcover": [float(x) for x in weather_data["cloudcover"].tolist()],
                "time": [str(x) for x in weather_data["time"].tolist()]
            }
        }

    def _calculate_frost_statistics(self, df):
        """Calcula estadísticas de riesgo de heladas"""
        total_hours = len(df)
        
        # Conteo por nivel de riesgo
        risk_counts = df['riesgo_helada'].value_counts().to_dict()
        
        # Convertir a tipos nativos
        for key in risk_counts:
            risk_counts[key] = int(risk_counts[key])
        
        # Porcentajes
        risk_percentages = {}
        for risk, count in risk_counts.items():
            risk_percentages[risk] = float(count / total_hours * 100)
        
        # Tipos de helada más comunes
        frost_types = df[df['riesgo_helada'] != 'sin_riesgo']['tipo_helada']
        if not frost_types.empty:
            # Aplanar listas de tipos
            all_types = []
            for types_list in frost_types:
                all_types.extend(types_list)
            frost_type_counts = pd.Series(all_types).value_counts().to_dict()
            frost_type_counts = {k: int(v) for k, v in frost_type_counts.items()}
        else:
            frost_type_counts = {}
        
        # Estadísticas de temperatura
        temp_stats = {
            "min_temperature": float(df['temperature_2m'].min()),
            "max_temperature": float(df['temperature_2m'].max()),
            "mean_temperature": float(df['temperature_2m'].mean()),
            "min_dew_point": float(df['punto_rocio'].min()),
            "max_dew_point": float(df['punto_rocio'].max()),
            "mean_dew_point": float(df['punto_rocio'].mean())
        }
        
        return {
            "total_hours_analyzed": int(total_hours),
            "risk_level_counts": risk_counts,
            "risk_level_percentages": risk_percentages,
            "frost_type_counts": frost_type_counts,
            "temperature_statistics": temp_stats
        }

    def _generate_frost_summary(self, df, statistics):
        """Genera un resumen de la predicción de heladas"""
        
        # Determinar nivel de riesgo general
        severe_hours = statistics["risk_level_counts"].get("severa", 0)
        moderate_hours = statistics["risk_level_counts"].get("moderada", 0)
        light_hours = statistics["risk_level_counts"].get("leve", 0)
        
        if severe_hours > 0:
            overall_risk = "MUY ALTO"
            risk_description = f"Condiciones de helada severa detectadas durante {severe_hours} horas"
        elif moderate_hours > 0:
            overall_risk = "ALTO"
            risk_description = f"Condiciones de helada moderada detectadas durante {moderate_hours} horas"
        elif light_hours > 0:
            overall_risk = "MODERADO"
            risk_description = f"Condiciones de helada leve detectadas durante {light_hours} horas"
        else:
            overall_risk = "BAJO"
            risk_description = "No se detectaron condiciones significativas de helada"
        
        # Períodos críticos
        critical_periods = []
        current_period = None
        
        for idx, row in df.iterrows():
            if row['riesgo_helada'] != 'sin_riesgo':
                time_str = str(row['time'])
                if current_period is None:
                    current_period = {
                        "start_time": time_str,
                        "risk_level": row['riesgo_helada'],
                        "frost_types": row['tipo_helada'],
                        "temperature": float(row['temperature_2m']),
                        "dew_point": float(row['punto_rocio'])
                    }
                else:
                    current_period["end_time"] = time_str
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
        if severe_hours > 0:
            recommendations.extend([
                "ALERTA MÁXIMA: Proteger cultivos sensibles con mantas térmicas",
                "Revisar sistemas de riego y calefacción",
                "Monitorear constantemente las condiciones"
            ])
        elif moderate_hours > 0:
            recommendations.extend([
                "Proteger plantas sensibles",
                "Considerar métodos de protección activa",
                "Evitar actividades agrícolas sensibles"
            ])
        elif light_hours > 0:
            recommendations.extend([
                "Vigilar plantas más sensibles",
                "Preparar métodos de protección preventiva"
            ])
        else:
            recommendations.append("Condiciones favorables, sin precauciones especiales necesarias")
        
        return {
            "overall_risk_level": overall_risk,
            "risk_description": risk_description,
            "critical_periods": critical_periods,
            "recommendations": recommendations,
            "dominant_frost_types": list(statistics["frost_type_counts"].keys())[:3] if statistics["frost_type_counts"] else []
        }

    def _format_hourly_data(self, df, limit=48):
        """Formatea datos horarios para la respuesta (limita a 48 horas)"""
        df_limited = df.head(limit)
        
        hourly_data = []
        for _, row in df_limited.iterrows():
            hourly_data.append({
                "time": str(row['time']),
                "temperature": float(row['temperature_2m']),
                "humidity": float(row['relativehumidity_2m']),
                "wind_speed": float(row['windspeed_10m']),
                "cloud_cover": float(row['cloudcover']),
                "dew_point": float(row['punto_rocio']),
                "frost_risk": str(row['riesgo_helada']),
                "frost_types": [str(t) for t in row['tipo_helada']],
                "risk_factors": [str(f) for f in row['factores_riesgo']]
            })
        
        return hourly_data

    def generar_reporte_heladas(self, df):
        """
        Genera un reporte detallado de las condiciones de helada
        """
        print("\nREPORTE DE RIESGO DE HELADAS")
        print("=" * 50)

        # Conteo de niveles de riesgo
        print("\nDistribución de niveles de riesgo:")
        print(df['riesgo_helada'].value_counts())

        # Tipos de helada detectados
        tipos_helada = df[df['riesgo_helada'] != 'sin_riesgo']['tipo_helada'].value_counts()
        if not tipos_helada.empty:
            print("\nTipos de helada detectados:")
            print(tipos_helada)

        # Estadísticas de temperatura y punto de rocío
        print("\nEstadísticas de temperatura:")
        print(f"Temperatura mínima: {df['temperature_2m'].min():.1f}°C")
        print(f"Punto de rocío mínimo: {df['punto_rocio'].min():.1f}°C")

        # Alertas de riesgo severo
        riesgo_severo = df[df['riesgo_helada'] == 'severa']
        if not riesgo_severo.empty:
            print("\n¡ALERTA! Se detectaron condiciones de helada severa:")
            print(f"Número de períodos con riesgo severo: {len(riesgo_severo)}")


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del predictor
    predictor = FrostRiskPredictor()
    
    # Ejemplo de predicción para una ciudad
    print("🌡️ SISTEMA DE PREDICCIÓN DE HELADAS")
    print("=" * 50)
    
    # Probar con una ciudad
    city = "Manizales"
    result = predictor.predict_frost_risk(city, days=3)
    
    if "error" not in result:
        print(f"\n📊 Predicción para {city}:")
        print(f"   Nivel de riesgo: {result['summary']['overall_risk_level']}")
        print(f"   Descripción: {result['summary']['risk_description']}")
        print(f"   Períodos críticos: {len(result['summary']['critical_periods'])}")
        
        if result['summary']['recommendations']:
            print("\n📋 Recomendaciones:")
            for rec in result['summary']['recommendations']:
                print(f"   • {rec}")
    else:
        print(f"❌ Error: {result['error']}")

