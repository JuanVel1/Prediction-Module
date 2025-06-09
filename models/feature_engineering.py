"""
Feature Engineering para Predicción de Deslizamientos de Tierra
"""

import pandas as pd
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

class LandslideFeatureEngineer:
    """Clase para generar características para predicción de deslizamientos"""
    
    def __init__(self, df):
        """
        Inicializar el feature engineer
        
        Args:
            df: DataFrame con datos de humedad del suelo
        """
        self.df = df.copy()
        self.prepare_base_data()
    
    def prepare_base_data(self):
        """Preparar datos base"""
        # Convertir time a datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['time']):
            self.df['time'] = pd.to_datetime(self.df['time'])
        
        # Crear coordenadas únicas para cada punto
        self.df['location_id'] = self.df['lat'].astype(str) + '_' + self.df['lon'].astype(str)
        
        # Ordenar por tiempo y ubicación
        self.df = self.df.sort_values(['location_id', 'time']).reset_index(drop=True)
        
        print(f"📊 Datos preparados: {self.df.shape}")
    
    def create_temporal_features(self):
        """Crear características temporales"""
        print("⏰ Creando características temporales...")
        
        # Features básicos de tiempo
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['day'] = self.df['time'].dt.day
        self.df['dayofyear'] = self.df['time'].dt.dayofyear
        self.df['week'] = self.df['time'].dt.isocalendar().week
        self.df['quarter'] = self.df['time'].dt.quarter
        
        # Features cíclicos (para capturar estacionalidad)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['dayofyear'] / 365)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['dayofyear'] / 365)
        
        # Features de lag temporal por ubicación
        for lag in [1, 3, 7, 14, 30]:
            self.df[f'sm_lag_{lag}'] = self.df.groupby('location_id')['sm'].shift(lag)
        
        # Rolling averages
        for window in [3, 7, 14, 30]:
            self.df[f'sm_rolling_mean_{window}'] = (
                self.df.groupby('location_id')['sm']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            self.df[f'sm_rolling_std_{window}'] = (
                self.df.groupby('location_id')['sm']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
        
        # Cambios y tendencias
        self.df['sm_change_1d'] = self.df.groupby('location_id')['sm'].diff(1)
        self.df['sm_change_7d'] = self.df.groupby('location_id')['sm'].diff(7)
        self.df['sm_change_30d'] = self.df.groupby('location_id')['sm'].diff(30)
        
        # Tendencia reciente (pendiente de regresión en ventana móvil)
        def calculate_trend(series):
            if len(series) < 2:
                return np.nan
            x = np.arange(len(series))
            return np.polyfit(x, series, 1)[0]
        
        self.df['sm_trend_7d'] = (
            self.df.groupby('location_id')['sm']
            .rolling(window=7, min_periods=2)
            .apply(calculate_trend)
            .reset_index(0, drop=True)
        )
        
        print("✅ Características temporales creadas")
    
    def create_moisture_indicators(self):
        """Crear indicadores específicos de humedad"""
        print("💧 Creando indicadores de humedad...")
        
        # Percentiles históricos por ubicación
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            self.df[f'sm_percentile_{p}'] = (
                self.df.groupby('location_id')['sm']
                .transform(lambda x: x.quantile(p/100))
            )
        
        # Posición relativa respecto a percentiles
        self.df['sm_above_p90'] = (self.df['sm'] > self.df['sm_percentile_90']).astype(int)
        self.df['sm_above_p95'] = (self.df['sm'] > self.df['sm_percentile_95']).astype(int)
        self.df['sm_above_p99'] = (self.df['sm'] > self.df['sm_percentile_99']).astype(int)
        
        # Días consecutivos de alta humedad
        self.df['sm_high_consecutive'] = (
            self.df.groupby('location_id')['sm_above_p90']
            .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
        )
        
        # Índice de saturación (normalizado por ubicación)
        self.df['saturation_index'] = (
            self.df.groupby('location_id')['sm']
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        )
        
        # Velocidad de acumulación de humedad
        for window in [7, 14, 30]:
            self.df[f'moisture_accumulation_{window}d'] = (
                self.df.groupby('location_id')['sm_change_1d']
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
            )
        
        # Anomalías respecto a la media histórica
        self.df['sm_mean_historical'] = (
            self.df.groupby('location_id')['sm'].transform('mean')
        )
        self.df['sm_std_historical'] = (
            self.df.groupby('location_id')['sm'].transform('std')
        )
        self.df['sm_anomaly'] = (
            (self.df['sm'] - self.df['sm_mean_historical']) / 
            (self.df['sm_std_historical'] + 1e-8)
        )
        
        print("✅ Indicadores de humedad creados")
    
    def create_spatial_features(self):
        """Crear características espaciales"""
        print("🗺️ Creando características espaciales...")
        
        # Clustering geográfico
        coords = self.df[['lat', 'lon']].drop_duplicates()
        n_clusters = min(50, len(coords) // 100)  # Ajustar según densidad
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            coords['geo_cluster'] = kmeans.fit_predict(coords[['lat', 'lon']])
            
            # Merge back to main dataframe
            self.df = self.df.merge(coords, on=['lat', 'lon'], how='left')
        else:
            self.df['geo_cluster'] = 0
        
        # Estadísticas por cluster geográfico
        cluster_stats = (
            self.df.groupby(['geo_cluster', 'time'])['sm']
            .agg(['mean', 'std', 'min', 'max'])
            .reset_index()
        )
        cluster_stats.columns = ['geo_cluster', 'time', 'cluster_sm_mean', 
                               'cluster_sm_std', 'cluster_sm_min', 'cluster_sm_max']
        
        self.df = self.df.merge(cluster_stats, on=['geo_cluster', 'time'], how='left')
        
        # Diferencia respecto a la media del cluster
        self.df['sm_vs_cluster'] = self.df['sm'] - self.df['cluster_sm_mean']
        
        # Gradientes espaciales (simplificado)
        # En una implementación real, se usarían métodos más sofisticados
        self.df['lat_rounded'] = np.round(self.df['lat'], 2)
        self.df['lon_rounded'] = np.round(self.df['lon'], 2)
        
        # Promedio en grilla cercana
        grid_stats = (
            self.df.groupby(['lat_rounded', 'lon_rounded', 'time'])['sm']
            .mean()
            .reset_index()
            .rename(columns={'sm': 'grid_sm_mean'})
        )
        
        self.df = self.df.merge(
            grid_stats, 
            on=['lat_rounded', 'lon_rounded', 'time'], 
            how='left'
        )
        
        print("✅ Características espaciales creadas")
    
    def create_risk_indicators(self):
        """Crear indicadores de riesgo específicos"""
        print("🚨 Creando indicadores de riesgo...")
        
        # Umbrales críticos de humedad
        critical_threshold = 0.6
        high_threshold = 0.5
        
        # Días por encima de umbrales
        self.df['days_above_critical'] = (
            (self.df['sm'] > critical_threshold).astype(int)
        )
        
        # Índice de riesgo compuesto
        self.df['composite_risk_index'] = np.random.random(len(self.df))  # Placeholder
        
        # Clasificación de riesgo
        self.df['risk_category'] = pd.cut(
            self.df['composite_risk_index'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Bajo', 'Medio', 'Alto', 'Crítico']
        )
        
        print("✅ Indicadores de riesgo creados")
    
    def create_interaction_features(self):
        """Crear características de interacción"""
        print("🔗 Creando características de interacción...")
        
        # Interacciones entre humedad y tiempo
        self.df['sm_x_month'] = self.df['sm'] * self.df['month']
        self.df['sm_x_quarter'] = self.df['sm'] * self.df['quarter']
        
        # Interacciones entre tendencias y anomalías
        self.df['trend_x_anomaly'] = self.df['sm_trend_7d'] * self.df['sm_anomaly']
        
        # Interacciones espaciales
        self.df['sm_x_lat'] = self.df['sm'] * self.df['lat']
        self.df['sm_x_elevation'] = self.df['sm'] * self.df.get('elevation', 0)  # Si hay datos de elevación
        
        print("✅ Características de interacción creadas")
    
    def engineer_all_features(self):
        """Ejecutar todo el pipeline de feature engineering"""
        print("🔧 Iniciando Feature Engineering completo...")
        
        # Ejecutar métodos principales
        self.create_temporal_features()
        self.create_moisture_indicators()
        self.create_spatial_features()
        self.create_risk_indicators()
        self.create_interaction_features()
        
        # Limpiar columnas auxiliares
        cols_to_drop = ['lat_rounded', 'lon_rounded']
        self.df = self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns])
        
        print(f"🎉 Feature Engineering completado!")
        print(f"📊 Características finales: {self.df.shape[1]} columnas")
        print(f"🔢 Observaciones: {len(self.df):,}")
        
        # Resumen de features creados
        new_features = [col for col in self.df.columns 
                       if col not in ['time', 'lat', 'lon', 'sm', 'location_id']]
        print(f"✨ Nuevas características: {len(new_features)}")
        
        return self.df
    
    def get_feature_importance_categories(self):
        """Obtener categorías de características para análisis"""
        categories = {
            'temporal': [col for col in self.df.columns if any(x in col for x in 
                        ['year', 'month', 'day', 'week', 'quarter', 'sin', 'cos'])],
            
            'lag_features': [col for col in self.df.columns if 'lag_' in col],
            
            'rolling_features': [col for col in self.df.columns if 'rolling_' in col],
            
            'moisture_indicators': [col for col in self.df.columns if any(x in col for x in 
                                  ['percentile_', 'above_p', 'saturation', 'accumulation'])],
            
            'spatial_features': [col for col in self.df.columns if any(x in col for x in 
                               ['cluster', 'grid_', 'vs_cluster'])],
            
            'risk_indicators': [col for col in self.df.columns if any(x in col for x in 
                              ['risk', 'critical', 'days_above'])],
            
            'interaction_features': [col for col in self.df.columns if '_x_' in col]
        }
        
        return categories

def main():
    """Función principal para testing"""
    print("🔧 Feature Engineering para Predicción de Deslizamientos")
    print("📝 Módulo listo para usar con datos procesados")
    
    return "Feature Engineering module ready"

if __name__ == '__main__':
    result = main() 