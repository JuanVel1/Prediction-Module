"""
Modelo de Predicción de Deslizamientos de Tierra
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class LandslidePredictor:
    """Modelo principal para predicción de deslizamientos de tierra"""
    
    def __init__(self):
        """Inicializar el predictor"""
        self.models = {}
        self.is_trained = False
        
    def prepare_features(self, df):
        """Preparar características para el modelo"""
        print("🔧 Preparando características...")
        
        # Características principales
        feature_cols = [
            'month', 'quarter', 'sm',
            'sm_lag_1', 'sm_lag_3', 'sm_lag_7',
            'sm_rolling_mean_7', 'composite_risk_index'
        ]
        
        # Filtrar solo columnas que existen
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Datos de características
        X = df[available_features].copy()
        X = X.fillna(X.mean())
        
        print(f"✅ Características preparadas: {X.shape[1]} columnas")
        return X, available_features
    
    def create_synthetic_targets(self, df):
        """Crear targets sintéticos para demostración"""
        print("🎯 Creando targets de demostración...")
        
        # Target basado en humedad alta y riesgo
        risk_score = (
            (df['sm'] > 0.6).astype(int) * 0.3 +
            (df['composite_risk_index'] > 0.7).astype(int) * 0.4 +
            np.random.random(len(df)) * 0.3
        )
        
        risk_binary = (risk_score > 0.5).astype(int)
        
        targets = {
            'landslide_probability': risk_score,
            'landslide_risk': risk_binary
        }
        
        print("✅ Targets creados")
        return targets
    
    def train_models(self, X, targets):
        """Entrenar modelos"""
        print("🚀 Entrenando modelos...")
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, targets['landslide_risk'], test_size=0.2, random_state=42
        )
        
        # Modelo de clasificación
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        self.models['classification'] = rf_clf
        
        # Evaluar
        y_pred = rf_clf.predict(X_test)
        accuracy = np.mean(y_test == y_pred)
        
        print(f"✅ Modelo entrenado - Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return {'accuracy': accuracy}
    
    def predict(self, X):
        """Hacer predicciones"""
        if not self.is_trained:
            raise ValueError("❌ Modelos no entrenados")
        
        predictions = {}
        
        if 'classification' in self.models:
            risk_class = self.models['classification'].predict(X)
            risk_prob = self.models['classification'].predict_proba(X)[:, 1]
            predictions['landslide_risk'] = risk_class
            predictions['risk_probability'] = risk_prob
        
        return predictions

def main():
    """Función principal"""
    print("🎯 Modelo de Predicción de Deslizamientos de Tierra")
    print("📝 Módulo listo para entrenar con datos procesados")
    
    return "Landslide Predictor module ready"

if __name__ == '__main__':
    result = main() 