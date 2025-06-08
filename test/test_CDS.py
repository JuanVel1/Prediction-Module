import cdsapi 
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener credenciales de CDS desde variables de entorno
CDS_URL = os.getenv('CDS_URL')
CDS_KEY = os.getenv('CDS_KEY')


# Verificar que las variables se cargaron correctamente
print("CDS_URL:", CDS_URL)
print("CDS_KEY:", CDS_KEY[:5] + "..." if CDS_KEY else "No encontrada")

if not CDS_KEY:
    raise ValueError("No se encontró la clave de CDS API. Por favor, verifica tu archivo .env")

# Configurar el cliente de CDS
c = cdsapi.Client(
    url=CDS_URL,
    key=CDS_KEY,
    verify=True,
    quiet=True
) 

c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': ['total_precipitation', 'volumetric_soil_water_layer_1'],
        'year': '2023',
        'month': '12',
        'day': '01',
        'time': '12:00',
        'format': 'netcdf',
    },
    'flood_data.nc'
).download()
