from classes.ValidationSystem import ValidationSystem

print("🔍 Probando sistema de validación...")

try:
    # Crear instancia del validador
    validator = ValidationSystem()
    
    # Generar reporte básico sin datos de API (para probar los modelos)
    print("📊 Evaluando modelos...")
    
    # Evaluar modelo TensorFlow
    tf_eval = validator.evaluate_tensorflow_model("models/surface_pressure.keras", "LSTM")
    print(f"🤖 TensorFlow LSTM: {tf_eval.get('overall_validity', 0)}% - {tf_eval.get('status', 'N/A')}")
    
    # Evaluar modelo XGBoost
    xgb_eval = validator.evaluate_xgboost_model("models/precipitation.pkl")
    print(f"🌧️ XGBoost: {xgb_eval.get('overall_validity', 0)}% - {xgb_eval.get('status', 'N/A')}")
    
    # Evaluar algunas clases de predicción
    print("\n⚙️ Evaluando clases de predicción...")
    classes_to_test = ["ElectricStorm", "Fog", "FrostRisk"]
    
    for class_name in classes_to_test:
        try:
            class_eval = validator.evaluate_prediction_class(class_name)
            print(f"🔌 {class_name}: {class_eval.get('overall_validity', 0)}% - {class_eval.get('status', 'N/A')}")
        except Exception as e:
            print(f"❌ {class_name}: Error - {str(e)[:50]}...")
    
    # Generar reporte completo (sin datos de API)
    print("\n📋 Generando reporte completo...")
    full_report = validator.generate_comprehensive_validation_report()
    
    print(f"\n✅ RESULTADO FINAL:")
    print(f"🎯 Puntuación general del sistema: {full_report.get('overall_system_score', 0)}%")
    print(f"🔧 Número de componentes evaluados: {len(full_report.get('models_evaluation', {})) + len(full_report.get('prediction_classes_evaluation', {}))}")
    
    recommendations = full_report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recomendaciones:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print(f"\n⏰ Validación completada: {full_report.get('validation_timestamp', 'N/A')}")
    print("\n🎉 ¡Sistema de validación funcionando correctamente!")
    
except Exception as e:
    print(f"❌ Error en validación: {e}")
    import traceback
    traceback.print_exc() 