import sys
sys.path.append('.')
import pickle

print('Intentando cargar modelo...')
try:
    with open('models/landslide_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('✅ Modelo cargado exitosamente')
    print(f'Tipo de modelo: {type(model)}')
    print(f'Atributos del modelo: {dir(model)[:10]}')
    
    # Probar predicción simple
    import numpy as np
    test_data = np.array([[1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    pred = model.predict(test_data)
    print(f'✅ Predicción de prueba: {pred}')
    
except ImportError as e:
    print(f'❌ Error de importación: {e}')
except Exception as e:
    print(f'❌ Error general: {e}') 