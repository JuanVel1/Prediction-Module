import requests_cache
import pandas as pd
import sys
import requests
import numpy as np
from pathlib import Path
import openmeteo_requests
from retry_requests import retry
# Importaciones opcionales para datos meteorológicos
try:
     
    METEO_AVAILABLE = True
except ImportError:
    METEO_AVAILABLE = False
    print("⚠️  Librerías meteorológicas no disponibles. Instalar con: pip install openmeteo-requests requests-cache retry-requests")

# Agregar el directorio padre al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

class FogPredictor:
    """Sistema de predicción de niebla"""
    
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
            self.df = pd.DataFrame()  # Initialize empty DataFrame instead of None
        
        # Umbrales para predicción de niebla
        self.delta_td_umbral = 2.0  # °C
        self.humedad_umbral = 95.0  # %
        self.viento_umbral = 5.0    # m/s
        
        # Calcular estadísticas del dataset si está disponible
        if self.df is not None:
            self._calcular_estadisticas()

    def get_coordinates(self, location: str):
        """Obtiene las coordenadas de una ubicación usando Nominatim"""
        headers = {
            "User-Agent": "FogPredictor/1.0 (prediction@example.com)",
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
        if self.df is None:
            return
        
        # Calcular predicciones de niebla para el dataset histórico
        df_con_niebla = self.calcular_prediccion_niebla(self.df)
        
        # Estadísticas generales
        total_registros = len(df_con_niebla)
        casos_niebla = df_con_niebla['niebla'].sum()
        porcentaje_niebla = (casos_niebla / total_registros) * 100
        
        print(f"📊 Estadísticas del dataset:")
        print(f"   Total de registros: {total_registros:,}")
        print(f"   Casos de niebla: {casos_niebla:,}")
        print(f"   Porcentaje de niebla: {porcentaje_niebla:.2f}%")

    def calcular_prediccion_niebla(self, df_input):
        """Calcular predicción de niebla basada en criterios meteorológicos"""
        df = df_input.copy()
        
        # Calcular diferencia entre temperatura y punto de rocío
        df['delta_Td'] = df['temperature_2m'] - df['dewpoint_2m']
        
        # Criterios principales para formación de niebla
        criterio_temperatura = df['delta_Td'] <= self.delta_td_umbral
        criterio_humedad = df['relativehumidity_2m'] >= self.humedad_umbral
        criterio_viento = df['windspeed_10m'] < self.viento_umbral
        
        # Predicción binaria de niebla
        df['niebla'] = criterio_temperatura & criterio_humedad & criterio_viento
        
        # Calcular probabilidad de niebla basada en múltiples factores
        # Normalizar cada criterio a un valor entre 0 y 1
        prob_temp = np.maximum(0, (self.delta_td_umbral - df['delta_Td']) / self.delta_td_umbral)
        prob_humedad = np.maximum(0, (df['relativehumidity_2m'] - self.humedad_umbral) / (100 - self.humedad_umbral))
        prob_viento = np.maximum(0, (self.viento_umbral - df['windspeed_10m']) / self.viento_umbral)
        
        # Combinar probabilidades (promedio ponderado)
        df['probabilidad_niebla'] = (
            prob_temp * 0.4 +      # 40% peso a la diferencia de temperatura
            prob_humedad * 0.35 +  # 35% peso a la humedad
            prob_viento * 0.25     # 25% peso al viento
        )
        
        # Limitar probabilidad entre 0 y 1
        df['probabilidad_niebla'] = np.clip(df['probabilidad_niebla'], 0, 1)
        
        # Clasificar intensidad de niebla
        df['intensidad_niebla'] = pd.cut(
            df['probabilidad_niebla'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Muy Baja', 'Baja', 'Moderada', 'Alta', 'Muy Alta'],
            include_lowest=True
        )
        
        # Condiciones adicionales favorables para niebla
        df['condiciones_ideales'] = (
            (df['delta_Td'] <= 1) &           # Muy poca diferencia T-Td
            (df['relativehumidity_2m'] >= 98) & # Humedad muy alta
            (df['windspeed_10m'] <= 2)         # Viento muy débil
        )
        
        return df

    def buscar_por_ciudad(self, ciudad: str, dias_futuro: int = 7):
        """Buscar datos meteorológicos y calcular riesgo de niebla para una ciudad"""
        print(f"🔍 Buscando datos para: {ciudad}")
        
        # Obtener coordenadas
        lat, lon = self.get_coordinates(ciudad)
        if lat is None or lon is None:
            return None
        
        print(f"📍 Coordenadas: {lat:.4f}, {lon:.4f}")
        
        # Obtener datos meteorológicos actuales/futuros
        datos_meteorologicos = self._obtener_datos_meteorologicos(lat, lon, dias_futuro)
        
        if datos_meteorologicos is None:
            return None
        
        # Calcular predicciones de niebla
        datos_con_niebla = self.calcular_prediccion_niebla(datos_meteorologicos)
        
        # Generar resumen
        resumen = self._generar_resumen(datos_con_niebla, ciudad)
        
        return {
            'ciudad': ciudad,
            'coordenadas': {'lat': lat, 'lon': lon},
            'datos_completos': datos_con_niebla,
            'resumen': resumen,
            'estadisticas': {
                'probabilidad_promedio': float(datos_con_niebla['probabilidad_niebla'].mean()),
                'probabilidad_maxima': float(datos_con_niebla['probabilidad_niebla'].max()),
                'delta_td_promedio': float(datos_con_niebla['delta_Td'].mean())
            }
        }

    def _obtener_datos_meteorologicos(self, lat: float, lon: float, dias: int = 7):
        """Obtener datos meteorológicos de la API"""
        try:
            if not METEO_AVAILABLE:
                print("❌ Librerías meteorológicas no disponibles")
                return None
            
            # Setup the Open-Meteo API client with cache and retry
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": [
                    "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
                    "surface_pressure", "cloudcover", "windspeed_10m", "visibility"
                ],
                "forecast_days": dias
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
                )
            }
            
            # Extraer variables
            variables = [
                "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
                "surface_pressure", "cloudcover", "windspeed_10m", "visibility"
            ]
            
            for i, var in enumerate(variables):
                try:
                    hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
                except:
                    # Si no está disponible, usar valores por defecto
                    if var == "visibility":
                        hourly_data[var] = np.full(len(hourly_data["time"]), 10000)  # 10km por defecto
                    else:
                        hourly_data[var] = np.full(len(hourly_data["time"]), 0)
            
            df = pd.DataFrame(data=hourly_data)
            print(f"✅ Datos meteorológicos obtenidos: {len(df)} registros")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos meteorológicos: {e}")
            return None

    def _generar_resumen(self, df, ciudad):
        """Generar resumen de predicción de niebla"""
        total_registros = len(df)
        casos_niebla = df['niebla'].sum()
        porcentaje_niebla = (casos_niebla / total_registros) * 100
        
        # Distribución de intensidades
        distribucion_intensidad = df['intensidad_niebla'].value_counts().to_dict()
        
        # Períodos con alta probabilidad de niebla
        df_alta_prob = df[df['probabilidad_niebla'] > 0.6]
        periodos_criticos = []
        
        if len(df_alta_prob) > 0:
            for _, row in df_alta_prob.iterrows():
                periodos_criticos.append({
                    'fecha': row['time'].strftime('%Y-%m-%d %H:%M'),
                    'probabilidad': float(round(row['probabilidad_niebla'], 3)),
                    'intensidad': str(row['intensidad_niebla']),
                    'condiciones': {
                        'delta_Td': float(round(row['delta_Td'], 2)),
                        'humedad': float(round(row['relativehumidity_2m'], 1)),
                        'viento': float(round(row['windspeed_10m'], 1)),
                        'condiciones_ideales': bool(row['condiciones_ideales'])
                    }
                })
        
        # Horas de mayor riesgo (agrupadas por hora del día)
        if 'time' in df.columns:
            df['hora'] = df['time'].dt.hour
            riesgo_por_hora = df.groupby('hora')['probabilidad_niebla'].mean().to_dict()
            horas_mayor_riesgo = sorted(riesgo_por_hora.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            horas_mayor_riesgo = []
        
        return {
            'total_registros': total_registros,
            'casos_niebla': int(casos_niebla),
            'porcentaje_niebla': round(porcentaje_niebla, 2),
            'distribucion_intensidad': distribucion_intensidad,
            'periodos_criticos': periodos_criticos[:10],  # Máximo 10 períodos
            'estadisticas_condiciones': {
                'delta_td_promedio': float(df['delta_Td'].mean()),
                'delta_td_minimo': float(df['delta_Td'].min()),
                'humedad_promedio': float(df['relativehumidity_2m'].mean()),
                'humedad_maxima': float(df['relativehumidity_2m'].max()),
                'viento_promedio': float(df['windspeed_10m'].mean()),
                'viento_minimo': float(df['windspeed_10m'].min())
            },
            'horas_mayor_riesgo': [(int(hora), float(round(prob, 3))) for hora, prob in horas_mayor_riesgo]
        }

    def imprimir_resumen(self, resultado):
        """Imprimir resumen formateado del análisis"""
        if resultado is None:
            print("❌ No hay datos para mostrar")
            return
        
        resumen = resultado['resumen']
        
        print("\n" + "="*60)
        print(f"🌫️  ANÁLISIS DE PREDICCIÓN DE NIEBLA - {resultado['ciudad'].upper()}")
        print("="*60)
        
        print(f"\n📍 Ubicación: {resultado['ciudad']}")
        print(f"🌐 Coordenadas: {resultado['coordenadas']['lat']:.4f}, {resultado['coordenadas']['lon']:.4f}")
        
        print(f"\n📊 RESUMEN GENERAL")
        print(f"   Total de registros analizados: {resumen['total_registros']}")
        print(f"   Casos de niebla predichos: {resumen['casos_niebla']}")
        print(f"   Porcentaje de niebla: {resumen['porcentaje_niebla']:.1f}%")
        
        print(f"\n🎯 ESTADÍSTICAS PRINCIPALES")
        stats = resultado['estadisticas']
        print(f"   Probabilidad promedio: {stats['probabilidad_promedio']:.3f}")
        print(f"   Probabilidad máxima: {stats['probabilidad_maxima']:.3f}")
        print(f"   Diferencia T-Td promedio: {stats['delta_td_promedio']:.2f}°C")
        
        print(f"\n🌫️  DISTRIBUCIÓN DE INTENSIDAD")
        for intensidad, cantidad in resumen['distribucion_intensidad'].items():
            print(f"   {intensidad}: {cantidad} registros")
        
        print(f"\n📈 CONDICIONES METEOROLÓGICAS")
        condiciones = resumen['estadisticas_condiciones']
        print(f"   Diferencia T-Td: {condiciones['delta_td_promedio']:.2f}°C (mín: {condiciones['delta_td_minimo']:.2f}°C)")
        print(f"   Humedad relativa: {condiciones['humedad_promedio']:.1f}% (máx: {condiciones['humedad_maxima']:.1f}%)")
        print(f"   Velocidad del viento: {condiciones['viento_promedio']:.1f} m/s (mín: {condiciones['viento_minimo']:.1f} m/s)")
        
        if resumen['horas_mayor_riesgo']:
            print(f"\n⏰ HORAS DE MAYOR RIESGO")
            for hora, prob in resumen['horas_mayor_riesgo']:
                print(f"   {hora:02d}:00 hrs - Probabilidad: {prob:.3f}")
        
        if resumen['periodos_criticos']:
            print(f"\n⚠️  PERÍODOS CRÍTICOS (primeros 5)")
            for i, periodo in enumerate(resumen['periodos_criticos'][:5], 1):
                print(f"   {i}. {periodo['fecha']} - {periodo['intensidad']} ({periodo['probabilidad']:.3f})")
                cond = periodo['condiciones']
                ideal = "✓" if cond['condiciones_ideales'] else "✗"
                print(f"      ΔT-Td: {cond['delta_Td']}°C, H: {cond['humedad']}%, V: {cond['viento']} m/s {ideal}")
        
        print("\n" + "="*60)

# Función principal para pruebas
def main():
    """Función principal para pruebas"""
    predictor = FogPredictor()
    
    # Probar con una ciudad
    ciudad_prueba = "Medellín, Colombia"
    resultado = predictor.buscar_por_ciudad(ciudad_prueba, dias_futuro=5)
    
    if resultado:
        predictor.imprimir_resumen(resultado)
    else:
        print("❌ No se pudieron obtener datos para la ciudad especificada")

if __name__ == "__main__":
    main()