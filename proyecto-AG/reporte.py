import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams

# Configuración global de estilo
rcParams.update({
    'font.size': 10,
    'font.weight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold'
})

def graficar_seleccion_torneo():
    """Genera el diagrama de selección por torneo"""
    plt.figure(figsize=(10, 6))
    
    # Crear grafo dirigido
    G = nx.DiGraph()
    
    # Nodos con posiciones
    nodes = {
        "Población": (0, 1),
        "Torneo 1": (1, 1.5),
        "Torneo 2": (1, 0.5),
        "Padre 1": (2, 1.5),
        "Padre 2": (2, 0.5)
    }
    
    # Añadir nodos y bordes
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    
    edges = [
        ("Población", "Torneo 1", "Muestra aleatoria"),
        ("Población", "Torneo 2", "Muestra aleatoria"),
        ("Torneo 1", "Padre 1", "Mejor fitness"),
        ("Torneo 2", "Padre 2", "Mejor fitness")
    ]
    
    for src, dst, label in edges:
        G.add_edge(src, dst, label=label)
    
    # Dibujar grafo
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=2500, 
            node_color='lightblue', font_size=10, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, 
                               edge_labels=nx.get_edge_attributes(G, 'label'),
                               font_size=9)
    
    plt.title("Selección por Torneo", pad=20)
    plt.savefig('diagrama_torneo.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_operador_cruza():
    """Genera el diagrama del operador de cruza"""
    plt.figure(figsize=(14, 6))
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.4)
    
    # Padre 1
    plt.subplot(1, 3, 1)
    plt.title("Padres", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.8, "Padre 1: [R1, R2, R3, R4]", 
            ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.6, "Padre 2: [R5, R6, R7, R8]", 
            ha='center', va='center', fontsize=11)
    
    # Cruza
    plt.subplot(1, 3, 2)
    plt.title("Cruza en punto aleatorio", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.85, "Punto de cruza = 2", 
            ha='center', va='center', fontsize=11)
    plt.plot([0.2, 0.8], [0.6, 0.6], 'r--', lw=2)
    plt.text(0.5, 0.7, "[R1, R2 | R3, R4]", 
            ha='center', va='center', bbox=dict(facecolor='lavender', alpha=0.5))
    plt.text(0.5, 0.4, "[R5, R6 | R7, R8]", 
            ha='center', va='center', bbox=dict(facecolor='mistyrose', alpha=0.5))
    
    # Resultado
    plt.subplot(1, 3, 3)
    plt.title("Hijos después de corrección", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.8, "Hijo 1: [R1, R2, R7, R8]", 
            ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.6, "Hijo 2: [R5, R6, R3, R4]", 
            ha='center', va='center', fontsize=11)
    plt.text(0.5, 0.4, "(R3 y R7 reemplazados)", 
            ha='center', va='center', fontsize=10, style='italic')
    
    plt.savefig('diagrama_cruza.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_operador_mutacion():
    """Genera el diagrama del operador de mutación"""
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
    
    # Antes
    plt.subplot(1, 2, 1)
    plt.title("Individuo original", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.6, "[R1, R2, R3, R4]", 
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='honeydew', edgecolor='green', boxstyle='round'))
    
    # Después
    plt.subplot(1, 2, 2)
    plt.title("Después de mutación", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.6, "[R1, R5, R3, R4]", 
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='lavenderblush', edgecolor='red', boxstyle='round'))
    plt.text(0.5, 0.3, "R2 → R5 (Posición 1)", 
            ha='center', va='center', color='darkred', fontsize=10)
    
    plt.savefig('diagrama_mutacion.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_estrategia_poda():
    """Genera el diagrama de la estrategia de poda"""
    plt.figure(figsize=(14, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
    
    # Población inicial
    plt.subplot(1, 3, 1)
    plt.title("Población + Descendencia", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.8, "Población actual", ha='center', va='center')
    plt.text(0.5, 0.7, "10 individuos", ha='center', va='center')
    plt.text(0.5, 0.6, "Descendencia", ha='center', va='center')
    plt.text(0.5, 0.5, "10 nuevos individuos", ha='center', va='center')
    plt.text(0.5, 0.4, "Total: 20 individuos", ha='center', va='center',
            bbox=dict(facecolor='aliceblue', alpha=0.7))
    
    # Proceso de selección
    plt.subplot(1, 3, 2)
    plt.title("Selección Elitista", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.8, "Top 50% (élite)", ha='center', va='center')
    plt.text(0.5, 0.7, "5 mejores individuos", ha='center', va='center')
    plt.text(0.5, 0.6, "25% aleatorios", ha='center', va='center')
    plt.text(0.5, 0.5, "3 individuos diversos", ha='center', va='center')
    
    # Resultado final
    plt.subplot(1, 3, 3)
    plt.title("Nueva Generación", pad=15)
    plt.axis('off')
    plt.text(0.5, 0.8, "Población final", ha='center', va='center')
    plt.text(0.5, 0.7, "10 individuos:", ha='center', va='center')
    plt.text(0.5, 0.6, "- 5 élite", ha='center', va='center')
    plt.text(0.5, 0.5, "- 3 aleatorios", ha='center', va='center')
    plt.text(0.5, 0.4, "- 2 completados", ha='center', va='center')
    plt.text(0.5, 0.3, "Mantiene diversidad", ha='center', va='center',
            fontsize=9, style='italic')
    
    plt.savefig('diagrama_poda.png', dpi=300, bbox_inches='tight')
    plt.close()

def generar_todas_graficas():
    """Genera las cuatro gráficas principales"""
    print("Generando gráficas para el reporte...")
    graficar_seleccion_torneo()
    graficar_operador_cruza()
    graficar_operador_mutacion()
    graficar_estrategia_poda()
    print("\nGráficas generadas exitosamente:")
    print("✅ diagrama_torneo.png - Diagrama de selección por torneo")
    print("✅ diagrama_cruza.png - Operador de cruza con corrección")
    print("✅ diagrama_mutacion.png - Operador de mutación")
    print("✅ diagrama_poda.png - Estrategia de selección de supervivientes")

if __name__ == "__main__":
    generar_todas_graficas()