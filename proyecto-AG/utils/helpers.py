from models.datos import Reactivo, Habilidad
from typing import Dict

def validar_restricciones(reactivos_data: Dict[str, Reactivo]):
    """Valida que las restricciones del algoritmo se cumplan."""
    print("Validando restricciones...")
    
    for reactivo_id, reactivo in reactivos_data.items():
        suma_pesos = sum(reactivo.peso_habilidades.values())
        print(f"Reactivo {reactivo_id}: Suma de pesos = {suma_pesos:.6f}")
        assert abs(suma_pesos - 1.0) < 1e-6, f"Los pesos del reactivo {reactivo_id} no suman 1"
    
    print("✓ Todas las restricciones se cumplen correctamente")

def crear_datos_ejemplo():
    """Crea datos de ejemplo para probar el algoritmo."""
    reactivos_data = {
        'R1': Reactivo('R1', ['H1', 'H3', 'H5'], {'H1': 0.4, 'H3': 0.3, 'H5': 0.3}),
        'R2': Reactivo('R2', ['H3', 'H4'], {'H3': 0.6, 'H4': 0.4}),
        'R3': Reactivo('R3', ['H1', 'H2', 'H4', 'H6'], {'H1': 0.25, 'H2': 0.25, 'H4': 0.25, 'H6': 0.25}),
        'R4': Reactivo('R4', ['H2', 'H5']),
        'R5': Reactivo('R5', ['H6']),
        'R6': Reactivo('R6', ['H1', 'H7', 'H8'], {'H1': 0.5, 'H7': 0.3, 'H8': 0.2}),
        'R7': Reactivo('R7', ['H2', 'H3', 'H7']),
        'R8': Reactivo('R8', ['H4', 'H5', 'H6', 'H8'], {'H4': 0.3, 'H5': 0.3, 'H6': 0.2, 'H8': 0.2}),
        'R9': Reactivo('R9', ['H5', 'H6'], {'H5': 0.7, 'H6': 0.3}),
        'R10': Reactivo('R10', ['H7', 'H8'], {'H7': 0.6, 'H8': 0.4})
    }
    
    habilidades_data = {
        'H1': Habilidad('H1', 0.9),
        'H2': Habilidad('H2', 1.0),
        'H3': Habilidad('H3', 0.5),
        'H4': Habilidad('H4', 0.0),
        'H5': Habilidad('H5', 0.0),
        'H6': Habilidad('H6', 0.0),
        'H7': Habilidad('H7', 0.6),
        'H8': Habilidad('H8', 0.8)
    }
    
    conteo_reactivos = {
        'R1': 2,
        'R2': 3,
        'R3': 0,
        'R4': 0,
        'R5': 0,
        'R6': 1,
        'R7': 0,
        'R8': 1,
        'R9': 1,
        'R10': 0
    }
    
    return reactivos_data, habilidades_data, conteo_reactivos, list(reactivos_data.keys())