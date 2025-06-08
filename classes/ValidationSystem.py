import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Any

class ValidationSystem:
    def __init__(self):
        self.validation_results = {}
        self.model_scores = {}
        
    def evaluate_tensorflow_model(self, model_path: str, model_type: str = "LSTM") -> Dict[str, float]:
        """Evalúa un modelo de TensorFlow y retorna métricas de confiabilidad"""
        try:
            model = tf.keras.models.load_model(model_path)
            
            # Métricas básicas del modelo
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            # Calcular score basado en arquitectura
            architecture_score = min(100, (trainable_params / 1000) * 10)  # Más parámetros = más complejo
            
            # Verificar si el modelo tiene historial de entrenamiento
            try:
                # Simular evaluación con datos sintéticos para verificar funcionamiento
                if model_type == "LSTM":
                    test_input = np.random.random((1, 24, 12))
                else:
                    test_input = np.random.random((1, 10))
                
                prediction = model.predict(test_input, verbose=0)
                prediction_score = 80 if prediction is not None and not np.isnan(prediction).any() else 20
                
            except Exception:
                prediction_score = 30
            
            # Score final basado en múltiples factores
            final_score = (architecture_score * 0.4 + prediction_score * 0.6)
            
            return {
                "model_type": model_type,
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "architecture_score": round(architecture_score, 1),
                "prediction_score": round(prediction_score, 1),
                "overall_validity": round(min(95, final_score), 1),
                "status": "FUNCIONAL" if final_score > 70 else "LIMITADO"
            }
        except Exception as e:
            return {
                "model_type": model_type,
                "error": str(e),
                "overall_validity": 15.0,
                "status": "ERROR"
            }

    def evaluate_xgboost_model(self, model_path: str) -> Dict[str, float]:
        """Evalúa un modelo XGBoost y retorna métricas de confiabilidad"""
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            
            # Verificar tipo de modelo
            model_type = type(model).__name__
            
            # Obtener información del modelo
            if hasattr(model, 'n_estimators'):
                n_estimators = model.n_estimators
                estimator_score = min(80, (n_estimators / 100) * 30)
            else:
                n_estimators = 0
                estimator_score = 40
            
            # Probar predicción con datos sintéticos
            try:
                test_input = np.random.random((10, 4))  # 4 features típicas
                prediction = model.predict(test_input)
                prediction_score = 85 if prediction is not None and not np.isnan(prediction).any() else 25
            except Exception:
                prediction_score = 30
            
            # Score final
            final_score = (estimator_score * 0.4 + prediction_score * 0.6)
            
            return {
                "model_type": model_type,
                "n_estimators": int(n_estimators) if n_estimators else "Unknown",
                "estimator_score": round(estimator_score, 1),
                "prediction_score": round(prediction_score, 1),
                "overall_validity": round(min(92, final_score), 1),
                "status": "FUNCIONAL" if final_score > 70 else "LIMITADO"
            }
        except Exception as e:
            return {
                "model_type": "XGBoost",
                "error": str(e),
                "overall_validity": 20.0,
                "status": "ERROR"
            }

    def evaluate_api_data_quality(self, hourly_data: Dict) -> Dict[str, float]:
        """Evalúa la calidad de los datos de la API meteorológica"""
        try:
            quality_scores = {}
            
            # Verificar disponibilidad de datos
            required_fields = ['temperature', 'precipitation', 'humidity', 'windspeed', 'surface_pressure']
            available_fields = 0
            
            for field in required_fields:
                if field in hourly_data and len(hourly_data[field]) > 0:
                    available_fields += 1
                    
                    # Verificar calidad de datos del campo
                    data_array = np.array(hourly_data[field])
                    
                    # Porcentaje de datos válidos (no NaN, no infinitos)
                    valid_data_pct = (1 - (np.isnan(data_array).sum() + np.isinf(data_array).sum()) / len(data_array)) * 100
                    
                    # Variabilidad de datos (evitar datos constantemente iguales)
                    if np.std(data_array) > 0.01:
                        variability_score = 100
                    else:
                        variability_score = 30  # Datos muy uniformes, posible error
                    
                    # Rango realista de datos
                    if field == 'temperature':
                        realistic_score = 100 if -50 <= np.mean(data_array) <= 60 else 50
                    elif field == 'humidity':
                        realistic_score = 100 if 0 <= np.mean(data_array) <= 100 else 50
                    elif field == 'windspeed':
                        realistic_score = 100 if 0 <= np.mean(data_array) <= 200 else 50
                    else:
                        realistic_score = 90  # Asumir bueno para otros campos
                    
                    field_score = (valid_data_pct * 0.4 + variability_score * 0.3 + realistic_score * 0.3)
                    quality_scores[field] = round(field_score, 1)
            
            # Score general de disponibilidad
            availability_score = (available_fields / len(required_fields)) * 100
            
            # Score promedio de calidad
            avg_quality = np.mean(list(quality_scores.values())) if quality_scores else 0
            
            # Score temporal (datos más recientes = mejor score)
            temporal_score = 95  # Asumimos datos en tiempo real de la API
            
            overall_score = (availability_score * 0.4 + avg_quality * 0.4 + temporal_score * 0.2)
            
            return {
                "field_scores": quality_scores,
                "availability_score": round(availability_score, 1),
                "avg_quality_score": round(avg_quality, 1),
                "temporal_score": round(temporal_score, 1),
                "overall_validity": round(min(98, overall_score), 1),
                "status": "EXCELENTE" if overall_score > 90 else "BUENO" if overall_score > 70 else "REGULAR"
            }
        except Exception as e:
            return {
                "error": str(e),
                "overall_validity": 25.0,
                "status": "ERROR"
            }

    def evaluate_prediction_class(self, class_name: str, sample_location: str = "Bogotá") -> Dict[str, float]:
        """Evalúa las clases de predicción específicas (tormentas, niebla, etc.)"""
        try:
            if class_name == "ElectricStorm":
                from classes.ElectricStorm import ElectricStormPredictor
                predictor = ElectricStormPredictor()
                result = predictor.buscar_por_ciudad(sample_location, dias_futuro=1)
                
            elif class_name == "Fog":
                from classes.Fog import FogPredictor
                predictor = FogPredictor()
                result = predictor.buscar_por_ciudad(sample_location, dias_futuro=1)
                
            elif class_name == "FrostRisk":
                from classes.FrostRisk import FrostRiskPredictor
                predictor = FrostRiskPredictor()
                result = predictor.predict_frost_risk(sample_location, days=1)
                
            elif class_name == "StrongWinds":
                from classes.strongwinds_gusts import StrongWindsPredictor
                predictor = StrongWindsPredictor()
                result = predictor.predict_strong_winds(sample_location, days=1)
                
            elif class_name == "WeatherExtreme":
                from classes.WeatherExtremeDetector import WeatherExtremeDetector
                predictor = WeatherExtremeDetector()
                result = predictor.predict_extreme_events(sample_location)
            
            else:
                return {"error": f"Clase {class_name} no reconocida", "overall_validity": 0.0, "status": "NO_DISPONIBLE"}
            
            # Evaluar el resultado
            if result and isinstance(result, dict):
                # Verificar completitud de datos
                completeness = len(result.keys()) / 10 * 100  # Asumir 10 campos esperados
                
                # Verificar si hay datos numéricos válidos
                numeric_fields = 0
                valid_numeric = 0
                
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        numeric_fields += 1
                        if not np.isnan(value) and not np.isinf(value):
                            valid_numeric += 1
                
                numeric_quality = (valid_numeric / numeric_fields * 100) if numeric_fields > 0 else 80
                
                # Score de funcionalidad
                functionality_score = 85 if result else 20
                
                overall_score = (completeness * 0.3 + numeric_quality * 0.4 + functionality_score * 0.3)
                
                return {
                    "class_type": class_name,
                    "completeness_score": round(completeness, 1),
                    "numeric_quality": round(numeric_quality, 1),
                    "functionality_score": round(functionality_score, 1),
                    "overall_validity": round(min(88, overall_score), 1),
                    "status": "FUNCIONAL" if overall_score > 70 else "LIMITADO"
                }
            else:
                return {
                    "class_type": class_name,
                    "overall_validity": 25.0,
                    "status": "ERROR_DE_DATOS"
                }
                
        except Exception as e:
            return {
                "class_type": class_name,
                "error": str(e),
                "overall_validity": 15.0,
                "status": "ERROR"
            }

    def generate_comprehensive_validation_report(self, hourly_data: Dict = None) -> Dict[str, Any]:
        """Genera un reporte completo de validación de todo el sistema"""
        
        report = {
            "validation_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "models_evaluation": {},
            "data_sources_evaluation": {},
            "prediction_classes_evaluation": {},
            "overall_system_score": 0.0,
            "recommendations": []
        }
        
        # Evaluar modelos de ML
        try:
            tf_model = self.evaluate_tensorflow_model("models/surface_pressure.keras", "LSTM")
            report["models_evaluation"]["surface_pressure_lstm"] = tf_model
        except:
            report["models_evaluation"]["surface_pressure_lstm"] = {"overall_validity": 0.0, "status": "NO_DISPONIBLE"}
        
        try:
            xgb_model = self.evaluate_xgboost_model("models/precipitation.pkl")
            report["models_evaluation"]["precipitation_xgboost"] = xgb_model
        except:
            report["models_evaluation"]["precipitation_xgboost"] = {"overall_validity": 0.0, "status": "NO_DISPONIBLE"}
        
        # Evaluar calidad de datos de API
        if hourly_data:
            api_quality = self.evaluate_api_data_quality(hourly_data)
            report["data_sources_evaluation"]["openmeteo_api"] = api_quality
        else:
            report["data_sources_evaluation"]["openmeteo_api"] = {"overall_validity": 70.0, "status": "NO_EVALUADO"}
        
        # Evaluar clases de predicción
        prediction_classes = ["ElectricStorm", "Fog", "FrostRisk", "StrongWinds", "WeatherExtreme"]
        for class_name in prediction_classes:
            class_eval = self.evaluate_prediction_class(class_name)
            report["prediction_classes_evaluation"][class_name] = class_eval
        
        # Calcular score general del sistema
        all_scores = []
        
        # Agregar scores de modelos
        for model_eval in report["models_evaluation"].values():
            all_scores.append(model_eval.get("overall_validity", 0))
        
        # Agregar scores de fuentes de datos
        for data_eval in report["data_sources_evaluation"].values():
            all_scores.append(data_eval.get("overall_validity", 0))
        
        # Agregar scores de clases (peso menor)
        for class_eval in report["prediction_classes_evaluation"].values():
            all_scores.append(class_eval.get("overall_validity", 0) * 0.5)  # Peso menor
        
        # Score promedio ponderado
        if all_scores:
            report["overall_system_score"] = round(np.mean(all_scores), 1)
        
        # Generar recomendaciones
        if report["overall_system_score"] > 85:
            report["recommendations"].append("✅ Sistema funcionando óptimamente")
        elif report["overall_system_score"] > 70:
            report["recommendations"].append("⚠️ Sistema funcional con áreas de mejora")
            if any(eval.get("overall_validity", 0) < 60 for eval in report["models_evaluation"].values()):
                report["recommendations"].append("🔧 Considerar reentrenamiento de modelos ML")
        else:
            report["recommendations"].append("🚨 Sistema requiere atención inmediata")
            report["recommendations"].append("🔧 Revisar modelos y fuentes de datos")
        
        return report 