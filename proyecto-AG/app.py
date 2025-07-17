from models.datos import Reactivo, Habilidad
from algoritmo.genetico import AlgoritmoGenetico
from visualizacion.graficos import VisualizadorResultados
from utils.configurador import ConfiguradorParametros
from utils.helpers import crear_datos_ejemplo

def crear_datos_ejemplo():
    """Crea datos de ejemplo para probar el algoritmo."""
    reactivos_data = {
        'R1': Reactivo('R1', ['H1', 'H3', 'H5'], {'H1': 0.4, 'H3': 0.3, 'H5': 0.3}),
        'R2': Reactivo('R2', ['H3', 'H4'], {'H3': 0.6, 'H4': 0.4}),
        # ... (resto de reactivos)
    }
    
    habilidades_data = {
        'H1': Habilidad('H1', 0.9),
        'H2': Habilidad('H2', 1.0),
    }
    
    conteo_reactivos = {
        'R1': 2,
        'R2': 3,
    }
    
    return reactivos_data, habilidades_data, conteo_reactivos, list(reactivos_data.keys())

def ejecutar_algoritmo():
    """Función principal para ejecutar el algoritmo genético."""
    print("Inicializando algoritmo genético...")
    
    # Crear datos de ejemplo
    reactivos_data, habilidades_data, conteo_reactivos, reactivos_alcanzables = crear_datos_ejemplo()
    
    # Configurar parámetros
    config = ConfiguradorParametros.recomendar_parametros(
        num_reactivos=len(reactivos_data),
        num_habilidades=len(habilidades_data),
        complejidad_problema="medio"
    )
    
    # Inicializar y ejecutar algoritmo genético
    ag = AlgoritmoGenetico(
        reactivos_alcanzables=reactivos_alcanzables,
        reactivos_data=reactivos_data,
        habilidades_data=habilidades_data,
        conteo_reactivos=conteo_reactivos,
        **config
    )
    
    ag.evolucionar()
    
    # Generar visualizaciones
    visualizador = VisualizadorResultados(ag)
    visualizador.generar_reporte_completo()
    
    return ag, visualizador

if __name__ == "__main__":
    print("="*60)
    print("ALGORITMO GENÉTICO PARA SELECCIÓN DE REACTIVOS EDUCATIVOS")
    print("="*60)
    
    algoritmo, visualizador = ejecutar_algoritmo()
    
    print("\n" + "="*60)
    print("EJECUCIÓN COMPLETADA")
    print("="*60)

# Ordenar los individuos por cantidad de genes del mayor al menor y seleccionar los individuos con genes de >1 
# Si hay una habilidad que tenga un valor (1) se descarta
# Los reactivos tienen que sumar 1
# Pero las habilidades no tienen que sumar 1, pueden ser 0 
# Los reactivos tienen que ser únicos, no puede haber duplicados

# H1=1 H1