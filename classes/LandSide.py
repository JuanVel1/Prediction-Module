import pandas as pd
import zipfile
import netCDF4
import xarray as xr
import os
import sys
import logging
from pathlib import Path
import numpy as np
import gc
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar el directorio padre al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

# Suprimir warnings para mejor rendimiento
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LandslidePredictionSystem:
    """Sistema completo de predicción de deslizamientos de tierra"""
    
    def __init__(self):
        """Inicializar el sistema de predicción"""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.model = None
        self.model_metadata = None
        self.weather_data: pd.DataFrame | None = None
        self.soil_moisture_data: pd.DataFrame | None = None
        
        # Configurar rutas
        self.model_path = self.project_root / 'models' / 'landslide_model.pkl'
        self.metadata_path = self.project_root / 'models' / 'trained' / 'model_metadata.pkl'
        self.weather_dataset = self.project_root / 'public' / 'datasets' / 'unido.xlsx'
        self.soil_data_path = self.project_root / 'public' / 'humedad_data_completo.csv.gz'
        
        logger.info("🏗️ Sistema de Predicción de Deslizamientos inicializado")
    
    def load_trained_model(self):
        """Cargar el modelo entrenado"""
        try:
            logger.info("🤖 Cargando modelo entrenado...")
            
            # Cargar modelo principal
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("✅ Modelo principal cargado")
            else:
                logger.error(f"❌ Modelo no encontrado en: {self.model_path}")
                return False
            
            # Cargar metadatos si existen
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.model_metadata = pickle.load(f)
                logger.info("✅ Metadatos del modelo cargados")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            return False
    
    def load_weather_data(self):
        """Cargar datos meteorológicos"""
        try:
            logger.info("🌤️ Cargando datos meteorológicos...")
            
            # Cargar datos del Excel
            self.weather_data = pd.read_excel(self.weather_dataset)
            
            # Asegurar que la columna time sea datetime
            if 'time' in self.weather_data.columns:
                self.weather_data['time'] = pd.to_datetime(self.weather_data['time'])
            
            logger.info(f"✅ Datos meteorológicos cargados: {self.weather_data.shape}")  # type: ignore
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos meteorológicos: {e}")
            return False
    
    def load_soil_moisture_data(self, sample_size=100000):
        """Cargar datos de humedad del suelo (muestra)"""
        try:
            logger.info("💧 Cargando datos de humedad del suelo...")
            
            # Cargar muestra de datos para eficiencia
            self.soil_moisture_data = pd.read_csv(
                self.soil_data_path, 
                compression='gzip',
                nrows=sample_size
            )
            
            # Convertir time a datetime si existe
            if 'time' in self.soil_moisture_data.columns:
                self.soil_moisture_data['time'] = pd.to_datetime(self.soil_moisture_data['time'])
            
            logger.info(f"✅ Datos de humedad cargados: {self.soil_moisture_data.shape}")  # type: ignore  # type: ignore
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos de humedad: {e}")
            return False
    
    def prepare_prediction_features(self, weather_sample, soil_sample):
        """Preparar características para predicción"""
        try:
            logger.info("🔧 Preparando características para predicción...")
            
            # Combinar datos por tiempo si es posible
            if 'time' in weather_sample.columns and 'time' in soil_sample.columns:
                # Unir por tiempo (tomar fechas más cercanas)
                weather_sample['date'] = weather_sample['time'].dt.date
                soil_sample['date'] = soil_sample['time'].dt.date
                
                # Usar outer join para obtener más datos
                combined = pd.merge(
                    weather_sample.groupby('date').first().reset_index(),
                    soil_sample.groupby('date').first().reset_index(),
                    on='date',
                    how='outer'
                )
                
                # Si hay pocos datos combinados, usar concatenación por índice
                if len(combined) < 100:
                    logger.info("⚠️ Pocos datos combinados por fecha, usando concatenación por índice...")
                    min_len = min(len(weather_sample), len(soil_sample))
                    combined = pd.concat([
                        weather_sample.iloc[:min_len].reset_index(drop=True),
                        soil_sample.iloc[:min_len].reset_index(drop=True)
                    ], axis=1)
            else:
                # Si no hay tiempo, combinar por índice
                min_len = min(len(weather_sample), len(soil_sample))
                combined = pd.concat([
                    weather_sample.iloc[:min_len].reset_index(drop=True),
                    soil_sample.iloc[:min_len].reset_index(drop=True)
                ], axis=1)
            
            # Crear características temporales
            if 'time_x' in combined.columns:
                combined['time'] = combined['time_x']
            elif 'time' in combined.columns:
                pass
            else:
                combined['time'] = pd.date_range('2024-01-01', periods=len(combined), freq='D')
            
            combined['month'] = combined['time'].dt.month
            combined['quarter'] = combined['time'].dt.quarter
            combined['day_of_year'] = combined['time'].dt.dayofyear
            
            # Crear características de humedad si existen
            if 'sm' in combined.columns:
                combined['sm_lag_1'] = combined['sm'].shift(1)
                combined['sm_lag_3'] = combined['sm'].shift(3)
                combined['sm_lag_7'] = combined['sm'].shift(7)
                combined['sm_rolling_mean_7'] = combined['sm'].rolling(7).mean()
            else:
                # Si no hay sm, crear una simulada basada en otros datos
                if 'relativehumidity_2m' in combined.columns and 'precipitation' in combined.columns:
                    combined['sm'] = (combined['relativehumidity_2m'] / 100 * 0.6 + 
                                    combined['precipitation'] / combined['precipitation'].max() * 0.4)
                    combined['sm_lag_1'] = combined['sm'].shift(1)
                    combined['sm_lag_3'] = combined['sm'].shift(3)
                    combined['sm_lag_7'] = combined['sm'].shift(7)
                    combined['sm_rolling_mean_7'] = combined['sm'].rolling(7).mean()
                else:
                    # Crear valores por defecto
                    combined['sm'] = 0.5
                    combined['sm_lag_1'] = 0.5
                    combined['sm_lag_3'] = 0.5
                    combined['sm_lag_7'] = 0.5
                    combined['sm_rolling_mean_7'] = 0.5
            
            # Crear índice de riesgo compuesto
            risk_factors = []
            
            if 'precipitation' in combined.columns:
                risk_factors.append(combined['precipitation'] / combined['precipitation'].max())
            if 'sm' in combined.columns:
                risk_factors.append(combined['sm'].tolist())  # type: ignore  # type: ignore
            if 'relativehumidity_2m' in combined.columns:
                risk_factors.append(combined['relativehumidity_2m'] / 100)
            
            if risk_factors:
                combined['composite_risk_index'] = np.mean(risk_factors, axis=0)
            else:
                combined['composite_risk_index'] = 0.5
            
            # Rellenar valores faltantes
            feature_columns = ['month', 'quarter', 'sm', 'sm_lag_1', 'sm_lag_3', 
                             'sm_lag_7', 'sm_rolling_mean_7', 'composite_risk_index']
            
            for col in feature_columns:
                if col in combined.columns:
                    combined[col] = combined[col].fillna(combined[col].mean())
            
            # Seleccionar características finales
            final_features = combined[feature_columns].copy()
            
            logger.info(f"✅ Características preparadas: {final_features.shape}")
            return final_features, combined
            
        except Exception as e:
            logger.error(f"❌ Error preparando características: {e}")
            return None, None
    
    def make_predictions(self, features_df):
        """Hacer predicciones de riesgo de deslizamientos"""
        try:
            logger.info("🎯 Generando predicciones...")
            
            if self.model is None:
                logger.error("❌ Modelo no cargado")
                return None
            
            # Hacer predicciones
            # El modelo devuelve un diccionario según nuestro diseño
            model_result = self.model.predict(features_df)
            
            if isinstance(model_result, dict) and 'risk_probability' in model_result:
                risk_probabilities = model_result['risk_probability']
            elif hasattr(self.model, 'predict_proba'):
                # Si es clasificador, obtener probabilidades
                probabilities = self.model.predict_proba(features_df)
                if probabilities.shape[1] > 1:
                    risk_probabilities = probabilities[:, 1]  # Probabilidad de riesgo alto
                else:
                    risk_probabilities = probabilities[:, 0]
            else:
                # Si es regresor, usar predicciones directas
                if isinstance(model_result, np.ndarray):
                    risk_probabilities = model_result
                else:
                    risk_probabilities = np.array([0.5] * len(features_df))
            
            # Clasificar niveles de riesgo
            risk_levels = []
            for prob in risk_probabilities:
                if prob < 0.2:
                    risk_levels.append("Muy Bajo")
                elif prob < 0.4:
                    risk_levels.append("Bajo")
                elif prob < 0.6:
                    risk_levels.append("Medio")
                elif prob < 0.8:
                    risk_levels.append("Alto")
                else:
                    risk_levels.append("Muy Alto")
            
            predictions = {
                'risk_probability': risk_probabilities,
                'risk_level': risk_levels,
                'binary_prediction': (risk_probabilities > 0.5).astype(int)
            }
            
            logger.info("✅ Predicciones generadas exitosamente")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error en predicciones: {e}")
            return None
    
    def generate_detailed_report(self, predictions, features_df, combined_data, location_name="Ubicación Ejemplo"):
        """Generar reporte detallado de predicciones"""
        try:
            logger.info("📊 Generando reporte detallado...")
            
            report = {
                'metadata': {
                    'location': location_name,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_accuracy': getattr(self.model_metadata, 'accuracy', 'N/A') if self.model_metadata else 'N/A',
                    'data_period': f"{combined_data['time'].min()} a {combined_data['time'].max()}",
                    'total_predictions': len(predictions['risk_probability'])
                },
                'summary_statistics': {
                    'avg_risk_probability': np.mean(predictions['risk_probability']),
                    'max_risk_probability': np.max(predictions['risk_probability']),
                    'min_risk_probability': np.min(predictions['risk_probability']),
                    'high_risk_days': np.sum(predictions['binary_prediction']),
                    'high_risk_percentage': np.mean(predictions['binary_prediction']) * 100
                },
                'risk_distribution': {
                    'Muy Bajo': np.sum([level == 'Muy Bajo' for level in predictions['risk_level']]),
                    'Bajo': np.sum([level == 'Bajo' for level in predictions['risk_level']]),
                    'Medio': np.sum([level == 'Medio' for level in predictions['risk_level']]),
                    'Alto': np.sum([level == 'Alto' for level in predictions['risk_level']]),
                    'Muy Alto': np.sum([level == 'Muy Alto' for level in predictions['risk_level']])
                },
                'feature_importance': {
                    'avg_soil_moisture': np.mean(features_df['sm']) if 'sm' in features_df.columns else 'N/A',
                    'avg_composite_risk': np.mean(features_df['composite_risk_index']) if 'composite_risk_index' in features_df.columns else 'N/A',
                    'seasonal_pattern': f"Q{features_df['quarter'].mode().iloc[0]}" if 'quarter' in features_df.columns else 'N/A'
                },
                'alerts_and_recommendations': []
            }
            
            # Generar alertas basadas en las predicciones
            if report['summary_statistics']['high_risk_percentage'] > 50:
                report['alerts_and_recommendations'].append({
                    'type': 'ALERTA CRÍTICA',
                    'message': f"Alto porcentaje de días con riesgo elevado ({report['summary_statistics']['high_risk_percentage']:.1f}%)"
                })
            
            if report['summary_statistics']['max_risk_probability'] > 0.8:
                report['alerts_and_recommendations'].append({
                    'type': 'ATENCIÓN',
                    'message': f"Probabilidad máxima de riesgo muy alta ({report['summary_statistics']['max_risk_probability']:.3f})"
                })
            
            # Identificar períodos de mayor riesgo
            high_risk_indices = np.where(predictions['risk_probability'] > 0.7)[0]
            if len(high_risk_indices) > 0:
                if 'time' in combined_data.columns:
                    high_risk_dates = combined_data['time'].iloc[high_risk_indices]
                    report['high_risk_periods'] = {
                        'dates': high_risk_dates.dt.strftime('%Y-%m-%d').tolist(),
                        'count': len(high_risk_indices)
                    }
            
            logger.info("✅ Reporte detallado generado")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte: {e}")
            return None
    
    def print_report(self, report):
        """Imprimir reporte detallado en consola"""
        print("\n" + "="*80)
        print("🏔️  REPORTE DETALLADO DE PREDICCIÓN DE DESLIZAMIENTOS DE TIERRA")
        print("="*80)
        
        # Metadatos
        print(f"\n📍 Ubicación: {report['metadata']['location']}")
        print(f"📅 Fecha de análisis: {report['metadata']['analysis_date']}")
        print(f"🎯 Precisión del modelo: {report['metadata']['model_accuracy']}")
        print(f"📊 Período de datos: {report['metadata']['data_period']}")
        print(f"📈 Total de predicciones: {report['metadata']['total_predictions']:,}")
        
        # Estadísticas de resumen
        print(f"\n📊 ESTADÍSTICAS DE RIESGO")
        print("-" * 40)
        print(f"• Probabilidad promedio: {report['summary_statistics']['avg_risk_probability']:.3f}")
        print(f"• Probabilidad máxima: {report['summary_statistics']['max_risk_probability']:.3f}")
        print(f"• Probabilidad mínima: {report['summary_statistics']['min_risk_probability']:.3f}")
        print(f"• Días de alto riesgo: {report['summary_statistics']['high_risk_days']:,}")
        print(f"• Porcentaje de alto riesgo: {report['summary_statistics']['high_risk_percentage']:.1f}%")
        
        # Distribución de riesgo
        print(f"\n🎨 DISTRIBUCIÓN DE NIVELES DE RIESGO")
        print("-" * 40)
        total = sum(report['risk_distribution'].values())
        for level, count in report['risk_distribution'].items():
            percentage = (count / total * 100) if total > 0 else 0
            emoji = {"Muy Bajo": "🟢", "Bajo": "🟡", "Medio": "🟠", "Alto": "🔴", "Muy Alto": "⚫"}
            print(f"{emoji.get(level, '⚪')} {level}: {count:,} casos ({percentage:.1f}%)")
        
        # Características importantes
        print(f"\n🔍 ANÁLISIS DE CARACTERÍSTICAS")
        print("-" * 40)
        print(f"• Humedad del suelo promedio: {report['feature_importance']['avg_soil_moisture']}")
        print(f"• Índice de riesgo compuesto: {report['feature_importance']['avg_composite_risk']}")
        print(f"• Patrón estacional dominante: {report['feature_importance']['seasonal_pattern']}")
        
        # Alertas y recomendaciones
        if report['alerts_and_recommendations']:
            print(f"\n⚠️  ALERTAS Y RECOMENDACIONES")
            print("-" * 40)
            for alert in report['alerts_and_recommendations']:
                print(f"🚨 {alert['type']}: {alert['message']}")
        
        # Períodos de alto riesgo
        if 'high_risk_periods' in report:
            print(f"\n📅 PERÍODOS DE ALTO RIESGO")
            print("-" * 40)
            print(f"• Total de períodos críticos: {report['high_risk_periods']['count']}")
            if len(report['high_risk_periods']['dates']) <= 10:
                print("• Fechas específicas:")
                for date in report['high_risk_periods']['dates'][:10]:
                    print(f"  - {date}")
            else:
                print("• Primeras 10 fechas críticas:")
                for date in report['high_risk_periods']['dates'][:10]:
                    print(f"  - {date}")
                print(f"  ... y {len(report['high_risk_periods']['dates']) - 10} más")
        
        print("\n" + "="*80)
        print("✅ Reporte completado")
        print("="*80)
    
    def run_prediction_analysis(self, location_name="Ciudad Ejemplo", sample_size=1000):
        """Ejecutar análisis completo de predicción"""
        logger.info("🚀 Iniciando análisis completo de predicción de deslizamientos")
        
        # 1. Cargar modelo
        logger.info("📋 Paso 1: Cargando modelo entrenado...")
        if not self.load_trained_model():
            logger.error("❌ Error en paso 1: No se pudo cargar el modelo")
            return None
        
        # 2. Cargar datos
        logger.info("📋 Paso 2: Cargando datos meteorológicos...")
        if not self.load_weather_data():
            logger.error("❌ Error en paso 2: No se pudieron cargar datos meteorológicos")
            return None
        
        logger.info("📋 Paso 3: Cargando datos de humedad del suelo...")
        if not self.load_soil_moisture_data(sample_size):
            logger.error("❌ Error en paso 3: No se pudieron cargar datos de humedad")
            return None
        
        # 3. Tomar muestras para análisis
        logger.info(f"📊 Datos disponibles - Meteorológicos: {len(self.weather_data):,}, Humedad: {len(self.soil_moisture_data):,}")
        
        weather_sample = self.weather_data.sample(n=min(sample_size//2, len(self.weather_data)), random_state=42)
        soil_sample = self.soil_moisture_data.sample(n=min(sample_size//2, len(self.soil_moisture_data)), random_state=42)
        
        logger.info(f"📊 Muestras tomadas - Meteorológicos: {len(weather_sample):,}, Humedad: {len(soil_sample):,}")
        
        # 4. Preparar características
        features_df, combined_data = self.prepare_prediction_features(weather_sample, soil_sample)
        
        if features_df is None:
            logger.error("❌ No se pudieron preparar las características")
            return None
        
        # 5. Hacer predicciones
        predictions = self.make_predictions(features_df)
        
        if predictions is None:
            logger.error("❌ No se pudieron generar predicciones")
            return None
        
        # 6. Generar reporte
        report = self.generate_detailed_report(predictions, features_df, combined_data, location_name)
        
        if report is None:
            logger.error("❌ No se pudo generar el reporte")
            return None
        
        # 7. Mostrar reporte
        self.print_report(report)
        
        # 8. Guardar resultados
        results = {
            'predictions': predictions,
            'features': features_df,
            'combined_data': combined_data,
            'report': report
        }
        
        # Guardar en archivo
        output_path = self.project_root / 'public' / f'prediction_results_{location_name.replace(" ", "_")}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"💾 Resultados guardados en: {output_path}")
        
        return results

def main():
    """Función principal"""
    print("🏔️  SISTEMA DE PREDICCIÓN DE DESLIZAMIENTOS DE TIERRA")
    print("="*60)
    
    # Crear instancia del sistema
    prediction_system = LandslidePredictionSystem()
    
    # Ejecutar análisis para una ciudad ejemplo
    ciudad_ejemplo = "Medellín, Colombia"
    
    logger.info(f"🎯 Iniciando análisis para: {ciudad_ejemplo}")
    
    results = prediction_system.run_prediction_analysis(
        location_name=ciudad_ejemplo,
        sample_size=10000  # Usar 10000 muestras para análisis más detallado
    )
    
    if results:
        print(f"\n🎉 ¡Análisis completado exitosamente para {ciudad_ejemplo}!")
        print(f"📊 Se analizaron {len(results['predictions']['risk_probability']):,} puntos de datos")
        print(f"⚠️  Días de alto riesgo detectados: {sum(results['predictions']['binary_prediction']):,}")
        print(f"📈 Probabilidad promedio de riesgo: {np.mean(results['predictions']['risk_probability']):.3f}")
    else:
        print("❌ Error en el análisis")
    
    return results

if __name__ == '__main__':
    results = main()
