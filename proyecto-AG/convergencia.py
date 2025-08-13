import matplotlib.pyplot as plt
import numpy as np
from algoritmo.genetico import AlgoritmoGenetico
from utils.helpers import crear_datos_ejemplo

def generar_grafica_convergencia():
    # Configuración común para todas las ejecuciones
    reactivos_data, habilidades_data, conteo_reactivos, reactivos_alcanzables = crear_datos_ejemplo()
    config = {
        'K': 3,
        'tamaño_poblacion': 20,
        'generaciones': 50,
        'tasa_mutacion': 0.15,
        'presion_seleccion': 0.7
    }
    
    plt.figure(figsize=(10, 6))
    
    # Ejecutar 3 veces el algoritmo y guardar los resultados
    for run in range(3):
        ag = AlgoritmoGenetico(
            reactivos_alcanzables=reactivos_alcanzables,
            reactivos_data=reactivos_data,
            habilidades_data=habilidades_data,
            conteo_reactivos=conteo_reactivos,
            **config
        )
        ag.evolucionar()
        
        # Graficar la evolución del fitness para esta ejecución
        plt.plot(ag.historial_fitness, label=f'Ejecución {run+1}', alpha=0.7, linewidth=2)
    
    # Configuración de la gráfica
    plt.title('Evolución del Fitness en Tres Ejecuciones Independientes', fontsize=14, pad=20)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig('convergencia.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfica de convergencia generada: convergencia.png")

if __name__ == "__main__":
    generar_grafica_convergencia()


