"""
Análisis Exploratorio de Datos (EDA) para Predicción de Deslizamientos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class SoilMoistureEDA:
    """Clase para análisis exploratorio de datos de humedad del suelo"""
    
    def __init__(self, data_path):
        """
        Inicializar el análisis
        
        Args:
            data_path: Ruta al archivo de datos procesado
        """
        self.data_path = Path(data_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Cargar datos procesados"""
        print("📊 Cargando datos de humedad del suelo...")
        
        if self.data_path.suffix == '.gz':
            self.df = pd.read_csv(self.data_path, compression='gzip')
        else:
            self.df = pd.read_csv(self.data_path)
        
        print(f"✅ Datos cargados: {self.df.shape}")
        print(f"📈 Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    def basic_info(self):
        """Información básica del dataset"""
        print("\n" + "="*50)
        print("📋 INFORMACIÓN BÁSICA")
        print("="*50)
        
        print(f"🔢 Dimensiones: {self.df.shape}")
        print(f"📊 Columnas: {list(self.df.columns)}")
        print(f"🗓️ Periodo temporal: {self.df['time'].min()} - {self.df['time'].max()}")
        
        # Información de valores nulos
        null_info = self.df.isnull().sum()
        if null_info.sum() > 0:
            print("\n⚠️ Valores nulos:")
            print(null_info[null_info > 0])
        else:
            print("\n✅ No hay valores nulos")
    
    def temporal_analysis(self):
        """Análisis temporal de la humedad"""
        print("\n" + "="*50)
        print("📅 ANÁLISIS TEMPORAL")
        print("="*50)
        
        # Convertir time a datetime si no lo está
        if not pd.api.types.is_datetime64_any_dtype(self.df['time']):
            self.df['time'] = pd.to_datetime(self.df['time'])
        
        # Crear features temporales
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['day'] = self.df['time'].dt.day
        self.df['dayofyear'] = self.df['time'].dt.dayofyear
        
        # Humedad promedio por día
        daily_avg = self.df.groupby('time')['sm'].mean().reset_index()
        
        # Gráfico temporal
        fig = px.line(daily_avg, x='time', y='sm', 
                     title='Evolución Temporal de la Humedad del Suelo',
                     labels={'sm': 'Humedad del Suelo', 'time': 'Fecha'})
        fig.show()
        
        return daily_avg
    
    def spatial_analysis(self):
        """Análisis espacial de la humedad"""
        print("\n" + "="*50)
        print("🗺️ ANÁLISIS ESPACIAL")
        print("="*50)
        
        # Humedad promedio por ubicación
        spatial_avg = self.df.groupby(['lat', 'lon'])['sm'].agg(['mean', 'std', 'count']).reset_index()
        
        # Mapa de calor de humedad
        fig = px.scatter_mapbox(
            spatial_avg.sample(min(10000, len(spatial_avg))), # Muestra para performance
            lat='lat', lon='lon', 
            color='mean',
            size='count',
            title='Distribución Espacial de Humedad del Suelo',
            mapbox_style="open-street-map",
            zoom=5,
            height=600
        )
        fig.show()
        
        return spatial_avg
    
    def statistical_summary(self):
        """Resumen estadístico detallado"""
        print("\n" + "="*50)
        print("📈 RESUMEN ESTADÍSTICO")
        print("="*50)
        
        # Variables numéricas de interés
        numeric_cols = ['sm', 'lat', 'lon']
        
        summary = self.df[numeric_cols].describe()
        print(summary)
        
        # Distribuciones
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histograma de humedad
        axes[0,0].hist(self.df['sm'].dropna(), bins=50, alpha=0.7, color='blue')
        axes[0,0].set_title('Distribución de Humedad del Suelo')
        axes[0,0].set_xlabel('Humedad')
        axes[0,0].set_ylabel('Frecuencia')
        
        # Box plot por mes
        monthly_data = [self.df[self.df['month'] == m]['sm'].dropna() for m in range(1, 13)]
        axes[0,1].boxplot(monthly_data, labels=range(1, 13))
        axes[0,1].set_title('Humedad por Mes')
        axes[0,1].set_xlabel('Mes')
        axes[0,1].set_ylabel('Humedad')
        
        # Mapa de calor de correlaciones
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,0])
            axes[1,0].set_title('Matriz de Correlaciones')
        
        # Tendencia temporal
        daily_avg = self.df.groupby('time')['sm'].mean()
        axes[1,1].plot(daily_avg.index, daily_avg.values)
        axes[1,1].set_title('Tendencia Temporal')
        axes[1,1].set_xlabel('Fecha')
        axes[1,1].set_ylabel('Humedad Promedio')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return summary
    
    def risk_analysis(self):
        """Análisis de patrones de riesgo"""
        print("\n" + "="*50)
        print("🚨 ANÁLISIS DE RIESGO")
        print("="*50)
        
        # Definir umbrales de riesgo (estos se ajustarán con datos históricos)
        self.df['risk_level'] = pd.cut(
            self.df['sm'], 
            bins=[0, 0.2, 0.4, 0.6, 1.0],
            labels=['Bajo', 'Medio', 'Alto', 'Crítico']
        )
        
        # Distribución de niveles de riesgo
        risk_dist = self.df['risk_level'].value_counts()
        print("Distribución de niveles de riesgo:")
        print(risk_dist)
        
        # Análisis temporal del riesgo
        risk_temporal = self.df.groupby(['time', 'risk_level']).size().unstack(fill_value=0)
        
        # Gráfico de niveles de riesgo en el tiempo
        fig = px.area(
            risk_temporal.reset_index(), 
            x='time', 
            y=['Bajo', 'Medio', 'Alto', 'Crítico'],
            title='Evolución de Niveles de Riesgo en el Tiempo'
        )
        fig.show()
        
        return risk_dist
    
    def generate_report(self, output_path='analysis/soil_moisture_report.html'):
        """Generar reporte completo"""
        print("\n" + "="*50)
        print("📄 GENERANDO REPORTE")
        print("="*50)
        
        # Ejecutar todos los análisis
        self.basic_info()
        daily_avg = self.temporal_analysis()
        spatial_avg = self.spatial_analysis()
        summary = self.statistical_summary()
        risk_dist = self.risk_analysis()
        
        print(f"\n✅ Análisis completado")
        print(f"📊 Total de observaciones: {len(self.df):,}")
        print(f"🗓️ Días analizados: {self.df['time'].nunique()}")
        print(f"📍 Ubicaciones únicas: {len(spatial_avg)}")
        
        return {
            'daily_avg': daily_avg,
            'spatial_avg': spatial_avg,
            'summary': summary,
            'risk_dist': risk_dist
        }

def main():
    """Función principal para ejecutar el EDA"""
    
    # Ruta al archivo de datos procesado
    data_path = '../public/humedad_data_completo.csv.gz'
    
    # Crear instancia del analizador
    eda = SoilMoistureEDA(data_path)
    
    # Generar reporte completo
    results = eda.generate_report()
    
    return results

if __name__ == '__main__':
    results = main() 