import requests
import json

def test_predictions_with_validation():
    """Prueba las predicciones con el sistema de validación incluido"""
    try:
        print("=== PRUEBA DE PREDICCIONES CON VALIDACIÓN ===\n")
        
        # Probar predicción con validación
        response = requests.post('http://localhost:8000/predict', json={'location': 'Bogotá'})
        data = response.json()
        
        # Mostrar información básica
        print(f"📍 Ubicación: {data.get('location', 'N/A')}")
        print(f"⏰ Generado: {data.get('generated_at', 'N/A')}")
        print()
        
        # Mostrar reporte de validación
        validation = data.get('validation_report', {})
        print("🔍 === REPORTE DE VALIDACIÓN ===")
        print(f"✅ Validez general del sistema: {validation.get('overall_system_validity', 'N/A')}")
        print(f"📊 Calidad de datos API: {validation.get('data_quality', 'N/A')}")
        print()
        
        print("🤖 Estado de modelos ML:")
        models_status = validation.get('models_status', {})
        for model_name, validity in models_status.items():
            status_icon = "✅" if float(validity.replace('%', '')) > 70 else "⚠️" if float(validity.replace('%', '')) > 50 else "❌"
            print(f"  {status_icon} {model_name}: {validity}")
        print()
        
        print(f"⚙️ Promedio clases de predicción: {validation.get('prediction_classes_avg', 'N/A')}")
        print()
        
        # Mostrar recomendaciones
        recommendations = validation.get('recommendations', [])
        if recommendations:
            print("💡 Recomendaciones del sistema:")
            for rec in recommendations:
                print(f"  • {rec}")
        print()
        
        # Mostrar resumen meteorológico
        summary = data.get('summary_24h', {})
        print("🌤️ === RESUMEN METEOROLÓGICO ===")
        temp_range = summary.get('temperature_range', {})
        print(f"🌡️ Temperatura: {temp_range.get('min', 0)}°C - {temp_range.get('max', 0)}°C")
        print(f"🌧️ Precipitación total: {summary.get('total_precipitation', 0):.1f} mm")
        print(f"💨 Viento máximo: {summary.get('max_windspeed', 0):.1f} m/s")
        print(f"⚠️ Horas de alto riesgo: {summary.get('high_risk_hours', 0)}")
        print()
        
        # Mostrar algunas predicciones horarias
        hourly = data.get('hourly_predictions', [])
        print("⏰ === MUESTRA DE PREDICCIONES HORARIAS ===")
        for i, hour in enumerate(hourly[:6]):
            risks = hour.get('risks', {})
            high_risks = [k.replace('_risk', '').upper() for k, v in risks.items() if v == 'ALTO']
            risk_indicator = "🔴" if high_risks else "🟢"
            
            print(f"{risk_indicator} {hour.get('hour', '')}: {hour.get('temperature', 0)}°C, "
                  f"💧{hour.get('humidity', 0)}%, 💨{hour.get('windspeed', 0)}m/s")
        
        if len(hourly) > 6:
            print(f"... y {len(hourly) - 6} horas más")
        
        print(f"\n📈 Total predicciones horarias: {len(hourly)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de predicciones: {e}")
        return False

def test_detailed_validation_report():
    """Prueba el endpoint de validación detallada"""
    try:
        print("\n" + "="*60)
        print("=== REPORTE DETALLADO DE VALIDACIÓN ===\n")
        
        response = requests.get('http://localhost:8000/validation')
        data = response.json()
        
        if data.get('status') == 'success':
            summary = data.get('summary', {})
            print(f"🎯 Puntuación general del sistema: {summary.get('overall_system_score', 'N/A')}")
            print(f"🔧 Componentes evaluados: {summary.get('total_components_evaluated', 'N/A')}")
            print(f"📊 Estado del sistema: {summary.get('system_status', 'N/A')}")
            print()
            
            # Detalles del reporte
            validation_report = data.get('validation_report', {})
            
            # Evaluación de modelos
            models_eval = validation_report.get('models_evaluation', {})
            if models_eval:
                print("🤖 === EVALUACIÓN DETALLADA DE MODELOS ===")
                for model_name, model_data in models_eval.items():
                    validity = model_data.get('overall_validity', 0)
                    status = model_data.get('status', 'UNKNOWN')
                    
                    status_icon = "✅" if validity > 80 else "⚠️" if validity > 60 else "❌"
                    print(f"{status_icon} {model_name}:")
                    print(f"   • Validez: {validity}%")
                    print(f"   • Estado: {status}")
                    
                    if 'total_params' in model_data:
                        print(f"   • Parámetros: {model_data['total_params']:,}")
                    if 'n_estimators' in model_data:
                        print(f"   • Estimadores: {model_data['n_estimators']}")
                    print()
            
            # Evaluación de fuentes de datos
            data_eval = validation_report.get('data_sources_evaluation', {})
            if data_eval:
                print("📡 === EVALUACIÓN DE FUENTES DE DATOS ===")
                for source_name, source_data in data_eval.items():
                    validity = source_data.get('overall_validity', 0)
                    status = source_data.get('status', 'UNKNOWN')
                    
                    status_icon = "✅" if validity > 90 else "⚠️" if validity > 70 else "❌"
                    print(f"{status_icon} {source_name}:")
                    print(f"   • Validez: {validity}%")
                    print(f"   • Estado: {status}")
                    
                    if 'availability_score' in source_data:
                        print(f"   • Disponibilidad: {source_data['availability_score']}%")
                    if 'avg_quality_score' in source_data:
                        print(f"   • Calidad promedio: {source_data['avg_quality_score']}%")
                    print()
            
            # Evaluación de clases de predicción
            classes_eval = validation_report.get('prediction_classes_evaluation', {})
            if classes_eval:
                print("⚙️ === EVALUACIÓN DE CLASES DE PREDICCIÓN ===")
                for class_name, class_data in classes_eval.items():
                    validity = class_data.get('overall_validity', 0)
                    status = class_data.get('status', 'UNKNOWN')
                    
                    status_icon = "✅" if validity > 80 else "⚠️" if validity > 60 else "❌"
                    print(f"{status_icon} {class_name}: {validity}% ({status})")
                print()
            
            return True
        else:
            print("❌ Error en el reporte de validación")
            return False
            
    except Exception as e:
        print(f"❌ Error obteniendo reporte detallado: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando pruebas del sistema de validación...")
    
    # Prueba 1: Predicciones con validación
    test1_result = test_predictions_with_validation()
    
    # Prueba 2: Reporte detallado
    test2_result = test_detailed_validation_report()
    
    print("\n" + "="*60)
    print("📋 === RESUMEN DE PRUEBAS ===")
    print(f"✅ Predicciones con validación: {'EXITOSO' if test1_result else 'FALLIDO'}")
    print(f"✅ Reporte detallado: {'EXITOSO' if test2_result else 'FALLIDO'}")
    
    if test1_result and test2_result:
        print("\n🎉 ¡Todas las pruebas exitosas! Sistema de validación funcionando correctamente.")
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revisar configuración del servidor.") 