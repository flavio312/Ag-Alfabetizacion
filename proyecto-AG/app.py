from models.datos import Reactivo, Habilidad
from algoritmo.genetico import AlgoritmoGenetico
from visualizacion.graficos import VisualizadorResultados
from utils.configurador import ConfiguradorParametros
from utils.helpers import crear_datos_ejemplo

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