import matplotlib.pyplot as plt
import numpy as np
from algoritmo.genetico import AlgoritmoGenetico
from utils.helpers import crear_datos_ejemplo
import pandas as pd
from matplotlib.gridspec import GridSpec

# Configuración de estilo actualizada
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 10,
    'font.weight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold'
})

def ejecutar_y_graficar_run(run_num, num_reactivos, num_habilidades, K, habilidades_aprobadas=None):
    """Ejecuta un run completo y genera todas las gráficas asociadas"""
    # Crear datos de ejemplo personalizados
    reactivos_data, habilidades_data, conteo_reactivos, reactivos_alcanzables = crear_datos_ejemplo()
    
    # Ajustar número de reactivos y habilidades
    reactivos_data = {k: reactivos_data[k] for k in list(reactivos_data.keys())[:num_reactivos]}
    habilidades_data = {k: habilidades_data[k] for k in list(habilidades_data.keys())[:num_habilidades]}
    reactivos_alcanzables = list(reactivos_data.keys())
    
    # Configurar habilidades aprobadas si se especifican
    if habilidades_aprobadas:
        for i, habilidad_id in enumerate(habilidades_data.keys()):
            habilidades_data[habilidad_id].calificacion = 0.8 if i < habilidades_aprobadas else 0.3
            habilidades_data[habilidad_id].aprobada = i < habilidades_aprobadas
    
    # Configurar parámetros del algoritmo
    config = {
        'K': K,
        'tamaño_poblacion': 20,
        'generaciones': 50,
        'tasa_mutacion': 0.15,
        'presion_seleccion': 0.7
    }
    
    # Ejecutar algoritmo genético
    ag = AlgoritmoGenetico(
        reactivos_alcanzables=reactivos_alcanzables,
        reactivos_data=reactivos_data,
        habilidades_data=habilidades_data,
        conteo_reactivos=conteo_reactivos,
        **config
    )
    ag.evolucionar()
    
    # Generar gráficas específicas para este run
    generar_grafica_fitness(ag, run_num)
    generar_grafica_metricas(ag, run_num)
    generar_grafica_mejora(ag, run_num)
    
    # Mostrar resultados en consola
    print(f"\nRUN {run_num} - RESULTADOS:")
    print(f"Reactivos seleccionados: {ag.mejor_individuo.genes}")
    print("Métricas:")
    for metrica, valor in ag.mejor_individuo.metricas.items():
        print(f"  {metrica}: {valor:.4f}")
    print(f"Fitness: {ag.mejor_individuo.fitness:.4f}")

def generar_grafica_fitness(ag, run_num):
    """Genera gráfica de evolución del fitness"""
    plt.figure(figsize=(10, 5))
    plt.plot(ag.historial_fitness, 'b-', linewidth=2)
    plt.title(f'RUN {run_num} - Evolución del Fitness', fontsize=12, pad=15)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'run{run_num}_fitness.png', dpi=300)
    plt.close()

def generar_grafica_metricas(ag, run_num):
    """Genera gráfica de comparación de métricas"""
    metricas = pd.DataFrame(ag.historial_metricas)
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig)
    
    # OBJ1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metricas['OBJ1'], 'g-')
    ax1.set_title('OBJ1: Habilidades No Aprobadas')
    ax1.grid(True, alpha=0.3)
    
    # OBJ2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metricas['OBJ2'], 'r-')
    ax2.set_title('OBJ2: Reactivos Ya Realizados')
    ax2.grid(True, alpha=0.3)
    
    # OBJ3
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metricas['OBJ3'], 'orange')
    ax3.set_title('OBJ3: Reactivos con Habilidades Aprobadas')
    ax3.grid(True, alpha=0.3)
    
    # OBJ4
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(metricas['OBJ4'], 'purple')
    ax4.set_title('OBJ4: Habilidades Involucradas')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'RUN {run_num} - Evolución de Métricas', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'run{run_num}_metricas.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_grafica_mejora(ag, run_num):
    """Genera gráfica de proyección de mejora"""
    simulacion = ag.mejor_individuo.simular_mejora()
    
    labels = ['Antes', 'Después']
    aprobadas = [simulacion['habilidades_aprobadas_antes'], 
                simulacion['habilidades_aprobadas_despues']]
    no_aprobadas = [len(ag.habilidades_data) - simulacion['habilidades_aprobadas_antes'],
                   len(ag.habilidades_data) - simulacion['habilidades_aprobadas_despues']]
    
    x = np.arange(2)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, aprobadas, width, label='Aprobadas', color='green')
    rects2 = ax.bar(x + width/2, no_aprobadas, width, label='No aprobadas', color='red')
    
    ax.set_title(f'RUN {run_num} - Proyección de Mejora en Habilidades', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.tight_layout()
    plt.savefig(f'run{run_num}_mejora.png', dpi=300)
    plt.close()

def main():
    print("EJECUTANDO SIMULACIONES PARA GENERAR RESULTADOS...")
    
    # Run 1
    print("\nEjecutando RUN 1...")
    ejecutar_y_graficar_run(
        run_num=1,
        num_reactivos=10,
        num_habilidades=8,
        K=3
    )
    
    # Run 2
    print("\nEjecutando RUN 2...")
    ejecutar_y_graficar_run(
        run_num=2,
        num_reactivos=15,
        num_habilidades=10,
        K=4,
        habilidades_aprobadas=3
    )
    
    # Run 3
    print("\nEjecutando RUN 3...")
    ejecutar_y_graficar_run(
        run_num=3,
        num_reactivos=20,
        num_habilidades=12,
        K=5
    )
    
    print("\nTODAS LAS GRÁFICAS HAN SIDO GENERADAS:")
    print("- run1_fitness.png: Evolución del fitness en Run 1")
    print("- run1_metricas.png: Métricas en Run 1")
    print("- run1_mejora.png: Mejora proyectada en Run 1")
    print("- run2_fitness.png: Evolución del fitness en Run 2")
    print("- run2_metricas.png: Métricas en Run 2")
    print("- run2_mejora.png: Mejora proyectada en Run 2")
    print("- run3_fitness.png: Evolución del fitness en Run 3")
    print("- run3_metricas.png: Métricas en Run 3")
    print("- run3_mejora.png: Mejora proyectada en Run 3")

if __name__ == "__main__":
    main()