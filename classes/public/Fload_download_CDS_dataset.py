import os
import cdsapi 
from dotenv import load_dotenv

load_dotenv()
CDS_URL = os.getenv('CDS_URL')
CDS_KEY = os.getenv('CDS_KEY')

if not CDS_KEY:
    raise ValueError("No se encontró la clave de CDS API. Por favor, verifica tu archivo .env")

c = cdsapi.Client(
    url=CDS_URL,
    key=CDS_KEY,
    verify=True,
    quiet=True
) 

# Configurar los ajustes para la descarga de datos
dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "relative_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ],
    "year": ["2024"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "06:00", "12:00",
        "18:00"
    ],
    "pressure_level": ["500", "700", "850"],
    "data_format": "grib",
    "download_format": "unarchived"
}
 
c.retrieve(dataset, request).download()
