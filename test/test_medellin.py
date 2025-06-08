import requests

response = requests.post('http://localhost:8000/predict', json={'location': 'Medellín'})
data = response.json()

print(f"Ciudad: {data['location']}")
print(f"Período: {data['forecast_period']}")
print(f"Coordenadas: lat={data['coordinates']['lat']}, lon={data['coordinates']['lon']}")
print(f"Horas con alto riesgo: {data['summary_24h']['high_risk_hours']}")
print(f"Temperatura: {data['summary_24h']['temperature_range']['min']}°C - {data['summary_24h']['temperature_range']['max']}°C")
print(f"Viento máximo: {data['summary_24h']['max_windspeed']} m/s")
print(f"Precipitación total: {data['summary_24h']['total_precipitation']} mm")
print(f"Total de predicciones horarias: {len(data['hourly_predictions'])}") 