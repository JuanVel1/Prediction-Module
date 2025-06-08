import requests
import json

try:
    response = requests.post('http://localhost:8000/predict', json={'location': 'Bogotá'})
    data = response.json()
    
    # Mostrar información básica
    print("=== PREDICCIÓN METEOROLÓGICA HORA POR HORA ===")
    print(f"Ubicación: {data.get('location', 'N/A')}")
    print(f"Coordenadas: {data.get('coordinates', {})}")
    print(f"Período: {data.get('forecast_period', 'N/A')}")
    print(f"Generado: {data.get('generated_at', 'N/A')}")
    print()
    
    # Mostrar resumen de 24 horas
    summary = data.get('summary_24h', {})
    print("=== RESUMEN 24 HORAS ===")
    temp_range = summary.get('temperature_range', {})
    print(f"Temperatura: {temp_range.get('min', 0)}°C - {temp_range.get('max', 0)}°C")
    print(f"Precipitación total: {summary.get('total_precipitation', 0):.1f} mm")
    print(f"Viento máximo: {summary.get('max_windspeed', 0):.1f} m/s")
    print(f"Ráfaga máxima: {summary.get('max_windgusts', 0):.1f} m/s")
    print(f"Horas de alto riesgo: {summary.get('high_risk_hours', 0)}")
    print()
    
    # Mostrar alertas
    alerts = data.get('general_alerts', [])
    print("=== ALERTAS ===")
    for alert in alerts:
        print(f"• {alert}")
    print()
    
    # Contar tipos de riesgo
    hourly = data.get('hourly_predictions', [])
    risk_counts = {'ALTO': 0, 'MEDIO': 0, 'BAJO': 0}
    
    print("=== TODAS LAS HORAS PREDICHAS ===")
    print("Hora".ljust(17) + "Temp".ljust(6) + "Lluvia".ljust(8) + "Viento".ljust(8) + "Humedad".ljust(9) + "Riesgos")
    print("-" * 70)
    
    for hour in hourly[:12]:  # Mostrar primeras 12 horas
        risks = hour.get('risks', {})
        high_risks = [k.replace('_risk', '').upper() for k, v in risks.items() if v == 'ALTO']
        medium_risks = [k.replace('_risk', '').upper() for k, v in risks.items() if v == 'MEDIO']
        
        # Contar riesgos
        for risk_level in risks.values():
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
        
        risk_str = ""
        if high_risks:
            risk_str = f"🔴 {','.join(high_risks)}"
        elif medium_risks:
            risk_str = f"🟡 {','.join(medium_risks)}"
        else:
            risk_str = "🟢 BAJO"
            
        hour_str = hour.get('hour', '').ljust(17)
        temp_str = f"{hour.get('temperature', 0)}°C".ljust(6)
        rain_str = f"{hour.get('precipitation', 0)}mm".ljust(8)
        wind_str = f"{hour.get('windspeed', 0)}m/s".ljust(8)
        humidity_str = f"{hour.get('humidity', 0)}%".ljust(9)
        
        print(f"{hour_str}{temp_str}{rain_str}{wind_str}{humidity_str}{risk_str}")
    
    if len(hourly) > 12:
        print(f"... y {len(hourly) - 12} horas más")
    
    print()
    print("=== ESTADÍSTICAS DE RIESGO ===")
    total_risk_instances = sum(risk_counts.values())
    if total_risk_instances > 0:
        for level, count in risk_counts.items():
            percentage = (count / total_risk_instances) * 100
            print(f"{level}: {count} instancias ({percentage:.1f}%)")
    
    print(f"\nTotal de predicciones horarias: {len(hourly)}")
    
except Exception as e:
    print(f"Error al conectar con el servidor: {e}")
    print("Asegúrate de que el servidor esté ejecutándose en http://localhost:8000") 