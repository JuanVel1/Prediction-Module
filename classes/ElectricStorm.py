import sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry
# Importaciones opcionales para datos meteorológicos
try:
    METEO_AVAILABLE = True
except ImportError:
    METEO_AVAILABLE = False
    print("⚠️  Librerías meteorológicas no disponibles. Instalar con: pip install openmeteo-requests requests-cache retry-requests")

# Agregar el directorio padre al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

class ElectricStormPredictor:
    """Sistema de predicción de tormentas eléctricas"""
    
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
            self.df = None
        
        # Umbrales para alertas (se calcularán con el dataset)
        self.K_umbral = None
        self.TT_umbral = None
        self.LI_umbral = None
        
        # Inicializar umbrales si hay datos
        if self.df is not None:
            self._calcular_umbrales()

    def get_coordinates(self, location: str):
        """Obtiene las coordenadas de una ubicación usando Nominatim"""
        headers = {
            "User-Agent": "ElectricStormPredictor/1.0 (prediction@example.com)",
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

    def calcular_indices_aproximados(self, df_input):
        """Calcular índices de inestabilidad atmosférica"""
        df = df_input.copy()
        
        # Constantes atmosféricas
        lapse_rate = 9.8  # °C/km (gradiente adiabático seco)
        R = 287.05  # J/(kg·K) - constante específica del aire seco
        g = 9.80665  # m/s² - aceleración gravitacional

        def altura_nivel(P0, P, T):
            """Calcular altura de un nivel de presión usando ecuación barométrica"""
            try:
                return (np.log(P0 / P) * R * (T + 273.15)) / g
            except:
                return np.nan

        # Presiones de los niveles (hPa)
        P0 = df['surface_pressure'] / 100  # Convertir Pa a hPa
        P_850 = 850
        P_700 = 700
        P_500 = 500

        # Alturas aproximadas para cada nivel
        h_850 = altura_nivel(P0, P_850, df['temperature_2m'])
        h_700 = altura_nivel(P0, P_700, df['temperature_2m'])
        h_500 = altura_nivel(P0, P_500, df['temperature_2m'])

        # Temperaturas aproximadas a cada nivel (°C)
        df['T_850'] = df['temperature_2m'] - lapse_rate * (h_850 / 1000)
        df['T_700'] = df['temperature_2m'] - lapse_rate * (h_700 / 1000)
        df['T_500'] = df['temperature_2m'] - lapse_rate * (h_500 / 1000)

        # Temperatura de rocío estimada en niveles superiores
        if 'dewpoint_2m' in df.columns:
            df['Td_850'] = df['dewpoint_2m'] - 1  # Ligera disminución con altura
            df['Td_700'] = df['dewpoint_2m'] - 3  # Mayor disminución
        else:
            # Aproximar usando humedad relativa
            df['Td_850'] = df['temperature_2m'] - ((100 - df['relativehumidity_2m']) / 5)
            df['Td_700'] = df['temperature_2m'] - ((100 - df['relativehumidity_2m']) / 4)

        # Temperatura de la parcela elevada adiabáticamente a 500 hPa
        df['T_parcela'] = df['temperature_2m'] - lapse_rate * (h_500 / 1000)

        # Cálculo de índices de inestabilidad
        # Índice K (George Index)
        df['K'] = (df['T_850'] - df['T_500']) + df['Td_850'] - (df['T_700'] - df['Td_700'])
        
        # Total Totals Index
        df['TT'] = (df['T_850'] + df['Td_850']) - 2 * df['T_500']
        
        # Lifted Index
        df['LI'] = df['T_500'] - df['T_parcela']

        # Limpiar valores infinitos o NaN
        for col in ['K', 'TT', 'LI']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].mean())

        return df

    def _calcular_umbrales(self):
        """Calcular umbrales basados en percentiles del dataset"""
        if self.df is None:
            return
        
        # Calcular índices para todo el dataset
        df_with_indices = self.calcular_indices_aproximados(self.df)
        
        # Definir umbrales basados en percentiles
        self.K_umbral = df_with_indices['K'].quantile(0.85)  # 85th percentile
        self.TT_umbral = df_with_indices['TT'].quantile(0.85)
        self.LI_umbral = df_with_indices['LI'].quantile(0.15)  # 15th percentile (valores bajos indican inestabilidad)
        
        print(f"📊 Umbrales calculados:")
        print(f"   K-Index: {self.K_umbral:.2f}")
        print(f"   Total Totals: {self.TT_umbral:.2f}")
        print(f"   Lifted Index: {self.LI_umbral:.2f}")

    def calcular_alerta_tormenta(self, df_input):
        """Calcular alertas de tormenta basadas en múltiples criterios"""
        df = df_input.copy()
        
        # Criterios principales basados en índices
        criterio_k = df['K'] > self.K_umbral if self.K_umbral else pd.Series([False] * len(df))
        criterio_tt = df['TT'] > self.TT_umbral if self.TT_umbral else pd.Series([False] * len(df))
        criterio_li = df['LI'] < self.LI_umbral if self.LI_umbral else pd.Series([False] * len(df))
        
        # Criterios adicionales basados en condiciones superficiales
        criterio_superficie = (
            (df['cloudcover'] > 70) & 
            (df.get('precipitation', 0) > 0) & 
            (df['windspeed_10m'] > 8)
        )
        
        # Combinación de criterios
        df['alerta_tormenta'] = (
            criterio_k | criterio_tt | criterio_li | criterio_superficie
        )
        
        # Nivel de riesgo
        risk_score = (
            (criterio_k.astype(int) * 0.3) +
            (criterio_tt.astype(int) * 0.3) +
            (criterio_li.astype(int) * 0.2) +
            (criterio_superficie.astype(int) * 0.2)
        )
        
        df['nivel_riesgo'] = pd.cut(
            risk_score, 
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Bajo', 'Moderado', 'Alto', 'Muy Alto'],
            include_lowest=True
        )
        
        return df

    def buscar_por_ciudad(self, ciudad: str, dias_futuro: int = 7):
        """Buscar datos meteorológicos y calcular riesgo de tormentas para una ciudad"""
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
        
        # Calcular índices de tormenta
        datos_con_indices = self.calcular_indices_aproximados(datos_meteorologicos)
        datos_con_alertas = self.calcular_alerta_tormenta(datos_con_indices)
        
        # Generar resumen
        resumen = self._generar_resumen(datos_con_alertas, ciudad)
        
        return {
            'ciudad': ciudad,
            'coordenadas': {'lat': lat, 'lon': lon},
            'datos_completos': datos_con_alertas,
            'resumen': resumen,
            'indices_promedio': {
                'K': float(datos_con_alertas['K'].mean()),
                'TT': float(datos_con_alertas['TT'].mean()),
                'LI': float(datos_con_alertas['LI'].mean())
            }
        }

    def _obtener_datos_meteorologicos(self, lat: float, lon: float, dias: int = 7):
        """Obtener datos meteorológicos de la API"""
        try:
            
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
                    "surface_pressure", "cloudcover", "windspeed_10m", "precipitation"
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
                "surface_pressure", "cloudcover", "windspeed_10m", "precipitation"
            ]
            
            for i, var in enumerate(variables):
                hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
            
            df = pd.DataFrame(data=hourly_data)
            print(f"✅ Datos meteorológicos obtenidos: {len(df)} registros")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos meteorológicos: {e}")
            return None

    def _generar_resumen(self, df, ciudad):
        """Generar resumen de riesgo de tormentas"""
        total_registros = len(df)
        alertas_activas = df['alerta_tormenta'].sum()
        porcentaje_riesgo = (alertas_activas / total_registros) * 100
        
        # Distribución de niveles de riesgo
        distribucion_riesgo = df['nivel_riesgo'].value_counts().to_dict()
        
        # Períodos de mayor riesgo
        df_riesgo_alto = df[df['nivel_riesgo'].isin(['Alto', 'Muy Alto'])]
        periodos_criticos = []
        
        if len(df_riesgo_alto) > 0:
            for _, row in df_riesgo_alto.iterrows():
                periodos_criticos.append({
                    'fecha': row['time'].strftime('%Y-%m-%d %H:%M'),
                    'nivel': str(row['nivel_riesgo']),
                    'indices': {
                        'K': float(round(row['K'], 2)),
                        'TT': float(round(row['TT'], 2)),
                        'LI': float(round(row['LI'], 2))
                    }
                })
        
        return {
            'total_registros': total_registros,
            'alertas_activas': int(alertas_activas),
            'porcentaje_riesgo': round(porcentaje_riesgo, 2),
            'distribucion_riesgo': distribucion_riesgo,
            'periodos_criticos': periodos_criticos[:10],  # Máximo 10 períodos
            'indices_maximos': {
                'K_max': float(df['K'].max()),
                'TT_max': float(df['TT'].max()),
                'LI_min': float(df['LI'].min())
            }
        }

    def imprimir_resumen(self, resultado):
        """Imprimir resumen formateado del análisis"""
        if resultado is None:
            print("❌ No hay datos para mostrar")
            return
        
        resumen = resultado['resumen']
        
        print("\n" + "="*60)
        print(f"⚡ ANÁLISIS DE TORMENTAS ELÉCTRICAS - {resultado['ciudad'].upper()}")
        print("="*60)
        
        print(f"\n📍 Ubicación: {resultado['ciudad']}")
        print(f"🌐 Coordenadas: {resultado['coordenadas']['lat']:.4f}, {resultado['coordenadas']['lon']:.4f}")
        
        print(f"\n📊 RESUMEN GENERAL")
        print(f"   Total de registros analizados: {resumen['total_registros']}")
        print(f"   Alertas de tormenta activas: {resumen['alertas_activas']}")
        print(f"   Porcentaje de riesgo: {resumen['porcentaje_riesgo']:.1f}%")
        
        print(f"\n🎯 ÍNDICES PROMEDIO")
        indices = resultado['indices_promedio']
        print(f"   K-Index: {indices['K']:.2f}")
        print(f"   Total Totals: {indices['TT']:.2f}")
        print(f"   Lifted Index: {indices['LI']:.2f}")
        
        print(f"\n🚨 DISTRIBUCIÓN DE RIESGO")
        for nivel, cantidad in resumen['distribucion_riesgo'].items():
            print(f"   {nivel}: {cantidad} registros")
        
        if resumen['periodos_criticos']:
            print(f"\n⚠️  PERÍODOS CRÍTICOS (primeros 5)")
            for i, periodo in enumerate(resumen['periodos_criticos'][:5], 1):
                print(f"   {i}. {periodo['fecha']} - Riesgo {periodo['nivel']}")
                print(f"      K={periodo['indices']['K']}, TT={periodo['indices']['TT']}, LI={periodo['indices']['LI']}")
        
        print("\n" + "="*60)

# Función principal para pruebas
def main():
    """Función principal para pruebas"""
    predictor = ElectricStormPredictor()
    
    # Probar con una ciudad
    ciudad_prueba = "Medellín, Colombia"
    resultado = predictor.buscar_por_ciudad(ciudad_prueba, dias_futuro=5)
    
    if resultado:
        predictor.imprimir_resumen(resultado)
    else:
        print("❌ No se pudieron obtener datos para la ciudad especificada")

if __name__ == "__main__":
    main()