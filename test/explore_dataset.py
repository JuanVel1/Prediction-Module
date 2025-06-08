import pandas as pd
import numpy as np

# Leer el dataset
print("📊 Cargando dataset unido.xlsx...")
df = pd.read_excel('public/datasets/unido.xlsx')

print(f"📏 Shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")
print("\n📄 Primeras filas:")
print(df.head())

# Buscar columnas relacionadas con ciudades
city_columns = [col for col in df.columns if any(word in col.lower() for word in ['ciudad', 'city', 'municipio', 'departamento', 'region'])]
print(f"\n🏙️ Columnas relacionadas con ciudades: {city_columns}")

if city_columns:
    for col in city_columns:
        print(f"\n{col} - Valores únicos (primeros 10):")
        print(df[col].unique()[:10])

# Información general
print(f"\n📊 Info del dataset:")
df.info() 