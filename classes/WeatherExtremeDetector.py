import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    from geopy.geocoders import Nominatim
    EXTERNAL_APIS_AVAILABLE = True
except ImportError:
    EXTERNAL_APIS_AVAILABLE = False

class WeatherExtremeDetector:
    def __init__(self):
        # Configurar cliente de Open-Meteo con cache y retry
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Configurar geocodificador
        self.geolocator = Nominatim(user_agent="extreme_weather_detector")
        
        # Configuración base que se ajustará dinámicamente
        self.config = {
            'heat_wave': {
                'min_duration_hours': 6,  # Reducido para mayor sensibilidad
                'humidity_threshold': 60,
                'risk_levels': {
                    'moderate': 30,  # Se ajustará dinámicamente
                    'high': 32,
                    'extreme': 35
                }
            },
            'cold_wave': {
                'min_duration_hours': 6,
                'risk_levels': {
                    'moderate': 5,   # Se ajustará dinámicamente
                    'high': 2,
                    'extreme': -2
                }
            }
        }

    def get_coordinates(self, city_name):
        """Obtiene las coordenadas de una ciudad usando Nominatim"""
        try:
            location = self.geolocator.geocode(city_name + ", Colombia")
            if location:
                return location.latitude, location.longitude
            else:
                # Intentar sin ", Colombia"
                location = self.geolocator.geocode(city_name)
                if location:
                    return location.latitude, location.longitude
                else:
                    raise ValueError(f"No se encontraron coordenadas para la ciudad: {city_name}")
        except Exception as e:
            raise ValueError(f"Error al obtener coordenadas: {str(e)}")

    def get_weather_data(self, latitude, longitude, days=7):
        """Obtiene datos meteorológicos de Open-Meteo"""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "apparent_temperature"],
                "forecast_days": days,
                "timezone": "America/Bogota"
            }
            
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Procesar datos horarios
            hourly = response.Hourly()
            hourly_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
            hourly_data["relativehumidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
            hourly_data["windspeed_10m"] = hourly.Variables(2).ValuesAsNumpy()
            hourly_data["apparent_temperature"] = hourly.Variables(3).ValuesAsNumpy()
            
            return pd.DataFrame(data=hourly_data)
            
        except Exception as e:
            raise ValueError(f"Error al obtener datos meteorológicos: {str(e)}")

    def calculate_heat_index(self, T, RH):
        """Calcula el índice de calor usando la fórmula de Rothfusz de la NOAA"""
        T_F = (T * 9/5) + 32

        if T_F < 80:
            HI = 0.5 * (T_F + 61.0 + ((T_F - 68.0) * 1.2) + (RH * 0.094))
        else:
            HI = -42.379 + 2.04901523*T_F + 10.14333127*RH \
                 - 0.22475541*T_F*RH - 0.00683783*T_F*T_F \
                 - 0.05481717*RH*RH + 0.00122874*T_F*T_F*RH \
                + 0.00085282*T_F*RH*RH - 0.00000199*T_F*T_F*RH*RH

            if RH < 13 and T_F >= 80 and T_F <= 112:
                adjustment = ((13-RH)/4) * np.sqrt((17-abs(T_F-95))/17)
                HI -= adjustment
            elif RH > 85 and T_F >= 80 and T_F <= 87:
                adjustment = ((RH-85)/10) * ((87-T_F)/5)
                HI += adjustment

        return (HI - 32) * 5/9

    def calculate_wind_chill(self, T, V):
        """Calcula la sensación térmica por viento"""
        T_F = (T * 9/5) + 32
        V_mph = V * 2.237

        if T_F <= 50 and V_mph >= 3:
            WC = 35.74 + 0.6215*T_F - 35.75 * \
                (V_mph**0.16) + 0.4275*T_F*(V_mph**0.16)
            return (WC - 32) * 5/9
        return T

    def _configure_thresholds(self, df):
        """Configura umbrales dinámicos basados en los datos"""
        temp_95th = df['temperature_2m'].quantile(0.95)
        temp_5th = df['temperature_2m'].quantile(0.05)
        
        self.config['heat_wave']['temp_threshold'] = temp_95th
        self.config['heat_wave']['risk_levels'] = {
            'moderate': temp_95th,
            'high': temp_95th + 2,
            'extreme': temp_95th + 4
        }
        
        self.config['cold_wave']['temp_threshold'] = temp_5th
        self.config['cold_wave']['risk_levels'] = {
            'moderate': temp_5th,
            'high': temp_5th - 2,
            'extreme': temp_5th - 4
        }

    def detect_extreme_events(self, df):
        """Detecta y clasifica olas de calor y frío con datos horarios"""
        df = df.copy()
        df = df.sort_values('time')

        # Configurar umbrales dinámicos
        self._configure_thresholds(df)

        # Calcular índices
        df['heat_index'] = df.apply(lambda x: self.calculate_heat_index(
            x['temperature_2m'], x['relativehumidity_2m']), axis=1)

        df['wind_chill'] = df.apply(lambda x: self.calculate_wind_chill(
            x['temperature_2m'], x['windspeed_10m']), axis=1)

        # Añadir hora del día
        df['hour'] = df['time'].dt.hour

        # Detectar eventos considerando la hora del día
        df['heat_wave'] = False
        df['cold_wave'] = False

        # Considerar horas diurnas para olas de calor (8:00 - 20:00)
        day_hours = (df['hour'] >= 8) & (df['hour'] <= 20)

        # Considerar horas nocturnas para olas de frío (20:00 - 8:00)
        night_hours = (df['hour'] < 8) | (df['hour'] > 20)

        heat_duration = self.config['heat_wave']['min_duration_hours']
        cold_duration = self.config['cold_wave']['min_duration_hours']
        
        for i in range(len(df) - max(heat_duration, cold_duration) + 1):
            # Ventana para ola de calor
            heat_window = df.iloc[i:i+heat_duration]
            
            # Detectar ola de calor durante el día
            if day_hours.iloc[i]:
                heat_conditions = (
                    (heat_window['temperature_2m'] >= self.config['heat_wave']['temp_threshold']) &
                    (heat_window['relativehumidity_2m'] >= self.config['heat_wave']['humidity_threshold'])
                )
                if heat_conditions.any():
                    df.iloc[i:i+heat_duration, df.columns.get_loc('heat_wave')] = True

            # Ventana para ola de frío
            cold_window = df.iloc[i:i+cold_duration]
            
            # Detectar ola de frío durante la noche
            if night_hours.iloc[i]:
                cold_conditions = (cold_window['temperature_2m'] <= self.config['cold_wave']['temp_threshold'])
                if cold_conditions.any():
                    df.iloc[i:i+cold_duration, df.columns.get_loc('cold_wave')] = True

        return df

    def _calculate_event_statistics(self, df):
        """Calcula estadísticas de eventos extremos"""
        try:
            # Estadísticas básicas
            heat_hours = int(df['heat_wave'].sum())
            cold_hours = int(df['cold_wave'].sum())
            
            # Análisis de períodos críticos
            heat_periods = []
            cold_periods = []
            
            # Identificar períodos continuos de olas de calor
            heat_changes = df['heat_wave'].astype(int).diff().fillna(0)
            heat_starts = df[heat_changes == 1].index.tolist()
            heat_ends = df[heat_changes == -1].index.tolist()
            
            if df['heat_wave'].iloc[0]:
                heat_starts = [0] + heat_starts
            if df['heat_wave'].iloc[-1]:
                heat_ends = heat_ends + [len(df)-1]
                
            for start, end in zip(heat_starts, heat_ends):
                period_data = df.iloc[start:end+1]
                if len(period_data) > 0:
                    heat_periods.append({
                        "inicio": period_data['time'].iloc[0].strftime('%Y-%m-%d %H:%M'),
                        "fin": period_data['time'].iloc[-1].strftime('%Y-%m-%d %H:%M'),
                        "duracion_horas": len(period_data),
                        "temperatura_maxima": float(period_data['temperature_2m'].max()),
                        "indice_calor_maximo": float(period_data['heat_index'].max())
                    })
            
            # Identificar períodos continuos de olas de frío
            cold_changes = df['cold_wave'].astype(int).diff().fillna(0)
            cold_starts = df[cold_changes == 1].index.tolist()
            cold_ends = df[cold_changes == -1].index.tolist()
            
            if df['cold_wave'].iloc[0]:
                cold_starts = [0] + cold_starts
            if df['cold_wave'].iloc[-1]:
                cold_ends = cold_ends + [len(df)-1]
                
            for start, end in zip(cold_starts, cold_ends):
                period_data = df.iloc[start:end+1]
                if len(period_data) > 0:
                    cold_periods.append({
                        "inicio": period_data['time'].iloc[0].strftime('%Y-%m-%d %H:%M'),
                        "fin": period_data['time'].iloc[-1].strftime('%Y-%m-%d %H:%M'),
                        "duracion_horas": len(period_data),
                        "temperatura_minima": float(period_data['temperature_2m'].min()),
                        "sensacion_termica_minima": float(period_data['wind_chill'].min())
                    })
            
            return {
                "total_horas_calor": heat_hours,
                "total_horas_frio": cold_hours,
                "periodos_calor": heat_periods,
                "periodos_frio": cold_periods,
                "umbral_calor": float(self.config['heat_wave']['temp_threshold']),
                "umbral_frio": float(self.config['cold_wave']['temp_threshold'])
            }
            
        except Exception as e:
            return {
                "total_horas_calor": 0,
                "total_horas_frio": 0,
                "periodos_calor": [],
                "periodos_frio": [],
                "error": f"Error en cálculo de estadísticas: {str(e)}"
            }

    def _determine_risk_level(self, df):
        """Determina el nivel de riesgo general"""
        heat_hours = df['heat_wave'].sum()
        cold_hours = df['cold_wave'].sum()
        
        total_hours = len(df)
        heat_percentage = (heat_hours / total_hours) * 100
        cold_percentage = (cold_hours / total_hours) * 100
        
        if heat_percentage > 20 or cold_percentage > 20:
            return "ALTO"
        elif heat_percentage > 10 or cold_percentage > 10:
            return "MODERADO"
        elif heat_percentage > 0 or cold_percentage > 0:
            return "BAJO"
        else:
            return "SIN_RIESGO"

    def _generate_recommendations(self, risk_level, stats):
        """Genera recomendaciones basadas en el nivel de riesgo"""
        recommendations = []
        
        if stats["total_horas_calor"] > 0:
            recommendations.extend([
                "🌡️ Se esperan períodos de calor extremo",
                "💧 Manténgase hidratado y busque sombra",
                "🏠 Evite actividades al aire libre en horas de mayor calor",
                "👥 Proteja a niños y adultos mayores"
            ])
        
        if stats["total_horas_frio"] > 0:
            recommendations.extend([
                "🥶 Se esperan períodos de frío extremo",
                "🧥 Use ropa abrigada en capas",
                "🏠 Mantenga espacios cerrados calefaccionados",
                "⚠️ Protéjase del viento para evitar mayor sensación de frío"
            ])
        
        if risk_level == "ALTO":
            recommendations.extend([
                "🚨 Condiciones extremas previstas",
                "📱 Manténgase informado sobre alertas meteorológicas",
                "🚗 Evite viajes innecesarios durante eventos extremos"
            ])
        
        return recommendations if recommendations else ["✅ No se esperan eventos climáticos extremos significativos"]

    def predict_extreme_events(self, city_name):
        """Método principal para predecir eventos extremos por ciudad"""
        try:
            # Obtener coordenadas
            latitude, longitude = self.get_coordinates(city_name)
            
            # Obtener datos meteorológicos
            df = self.get_weather_data(latitude, longitude)
            
            # Detectar eventos extremos
            df_with_events = self.detect_extreme_events(df)
            
            # Calcular estadísticas
            stats = self._calculate_event_statistics(df_with_events)
            
            # Determinar nivel de riesgo
            risk_level = self._determine_risk_level(df_with_events)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(risk_level, stats)
            
            return {
                "ciudad": city_name,
                "coordenadas": {"latitud": latitude, "longitud": longitude},
                "nivel_riesgo": risk_level,
                "periodo_analisis": {
                    "inicio": df['time'].min().strftime('%Y-%m-%d %H:%M'),
                    "fin": df['time'].max().strftime('%Y-%m-%d %H:%M'),
                    "total_horas": len(df)
                },
                "estadisticas": stats,
                "recomendaciones": recommendations,
                "fecha_prediccion": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                "ciudad": city_name,
                "error": f"Error en predicción de eventos extremos: {str(e)}",
                "nivel_riesgo": "DESCONOCIDO",
                "fecha_prediccion": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def plot_events(self, df):
        """Genera visualizaciones de los eventos extremos con datos horarios"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            # Gráfico de temperatura y eventos de calor
            ax1.plot(df['time'], df['temperature_2m'],
                     'r-', label='Temperatura', alpha=0.7)
            ax1.fill_between(df['time'], df['temperature_2m'],
                             where=df['heat_wave'],
                             color='red', alpha=0.3, label='Ola de Calor')
            ax1.axhline(y=self.config['heat_wave']['temp_threshold'],
                        color='r', linestyle=':', label=f'Umbral de Calor ({self.config["heat_wave"]["temp_threshold"]:.1f}°C)')
            ax1.set_title('Eventos de Calor Extremo')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Gráfico de temperatura y eventos de frío
            ax2.plot(df['time'], df['temperature_2m'],
                     'b-', label='Temperatura', alpha=0.7)
            ax2.fill_between(df['time'], df['temperature_2m'],
                             where=df['cold_wave'],
                             color='blue', alpha=0.3, label='Ola de Frío')
            ax2.axhline(y=self.config['cold_wave']['temp_threshold'],
                        color='b', linestyle=':', label=f'Umbral de Frío ({self.config["cold_wave"]["temp_threshold"]:.1f}°C)')
            ax2.set_title('Eventos de Frío Extremo')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️  No se pueden mostrar gráficos: {e}")

        # Análisis estadístico de los eventos
        print("\nAnálisis de Eventos Extremos:")
        print(f"Umbral de calor: {self.config['heat_wave']['temp_threshold']:.1f}°C")
        print(f"Umbral de frío: {self.config['cold_wave']['temp_threshold']:.1f}°C")
        print(f"Total de horas con ola de calor: {df['heat_wave'].sum()}")
        print(f"Total de horas con ola de frío: {df['cold_wave'].sum()}")

        # Análisis por día
        try:
            daily_events = df.groupby(df['time'].dt.date).agg({
                'heat_wave': 'sum',
                'cold_wave': 'sum',
                'temperature_2m': ['min', 'max', 'mean']
            })

            print("\nDías con eventos extremos:")
            event_days = daily_events[(daily_events['heat_wave']['sum'] > 0) |
                                      (daily_events['cold_wave']['sum'] > 0)]
            if not event_days.empty:
                print(event_days)
            else:
                print("No se detectaron días con eventos extremos")
        except Exception as e:
            print(f"⚠️  Error en análisis diario: {e}")


# Código principal
if __name__ == "__main__":
    print("🌡️ DETECTOR DE EVENTOS CLIMÁTICOS EXTREMOS")
    print("=" * 50)
    
    # Probar con una ciudad
    detector = WeatherExtremeDetector()
    result = detector.predict_extreme_events("Cali")
    
    print("Resultado de la predicción:")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))