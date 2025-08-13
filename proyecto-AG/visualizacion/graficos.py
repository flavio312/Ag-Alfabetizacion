import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Optional, List, Tuple
from models.individuo import Individuo
import networkx as nx
from matplotlib.patches import Rectangle

class VisualizadorResultados:
    """Genera visualizaciones para los resultados del algoritmo genético."""
    
    def __init__(self, algoritmo):
        self.algoritmo = algoritmo
        plt.style.use('seaborn-v0_8')
    
    def graficar_evolucion_fitness(self):
        """Gráfica la evolución del fitness."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.algoritmo.historial_fitness, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title('Evolución del Fitness', fontsize=14, fontweight='bold')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if len(self.algoritmo.historial_fitness) > 10:
            ultimas_fitness = self.algoritmo.historial_fitness[-10:]
            plt.plot(range(len(self.algoritmo.historial_fitness)-10, len(self.algoritmo.historial_fitness)), 
                    ultimas_fitness, 'r-', linewidth=2, marker='s', markersize=5)
            plt.title('Últimas 10 Generaciones', fontsize=14, fontweight='bold')
            plt.xlabel('Generación')
            plt.ylabel('Fitness')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def graficar_metricas_objetivos(self):
        """Gráfica la evolución de las métricas de los objetivos."""
        if not self.algoritmo.historial_metricas:
            return
        
        metricas_df = pd.DataFrame(self.algoritmo.historial_metricas)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(metricas_df['OBJ1'], 'g-', linewidth=2, marker='o')
        axes[0, 0].set_title('OBJ1: Habilidades No Aprobadas', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(metricas_df['OBJ2'], 'r-', linewidth=2, marker='s')
        axes[0, 1].set_title('OBJ2: Reactivos Ya Realizados', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(metricas_df['OBJ3'], 'orange', linewidth=2, marker='^')
        axes[1, 0].set_title('OBJ3: Reactivos con Habilidades Aprobadas', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(metricas_df['OBJ4'], 'purple', linewidth=2, marker='d')
        axes[1, 1].set_title('OBJ4: Habilidades Involucradas', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def graficar_estado_habilidades(self):
        """Gráfica el estado actual de las habilidades."""
        habilidades_data = list(self.algoritmo.habilidades_data.items())
        nombres = [h[0] for h in habilidades_data]
        calificaciones = [h[1].calificacion for h in habilidades_data]
        aprobadas = [h[1].aprobada for h in habilidades_data]
        
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        colores = ['green' if aprobada else 'red' for aprobada in aprobadas]
        bars = plt.bar(nombres, calificaciones, color=colores, alpha=0.7)
        plt.axhline(y=0.7, color='black', linestyle='--', alpha=0.8, label='Umbral de Aprobación')
        plt.title('Estado de Habilidades del Estudiante', fontsize=16, fontweight='bold')
        plt.ylabel('Calificación')
        plt.ylim(0, 1)
        plt.legend()
        plt.xticks(rotation=45)
        
        for bar, calificacion in zip(bars, calificaciones):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{calificacion:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(2, 1, 2)
        aprobadas_count = sum(aprobadas)
        no_aprobadas_count = len(aprobadas) - aprobadas_count
        
        plt.pie([aprobadas_count, no_aprobadas_count], 
                labels=['Aprobadas', 'No Aprobadas'], 
                colors=['lightgreen', 'lightcoral'], 
                autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05), shadow=True)
        plt.title('Distribución de Habilidades', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def graficar_analisis_reactivos_seleccionados(self):
        """Analiza y gráfica los reactivos seleccionados."""
        if not self.algoritmo.mejor_individuo:
            return
        
        mejor = self.algoritmo.mejor_individuo
        reactivos_seleccionados = mejor.genes
        nombres_reactivos = []
        habilidades_por_reactivo = []
        
        for reactivo_id in reactivos_seleccionados:
            reactivo = self.algoritmo.reactivos_data[reactivo_id]
            nombres_reactivos.append(reactivo_id)
            habilidades_por_reactivo.append(len(reactivo.habilidades))
        
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar(nombres_reactivos, habilidades_por_reactivo, color='skyblue', alpha=0.8)
        plt.title('Habilidades por Reactivo Seleccionado', fontweight='bold')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        todos_los_pesos = []
        for reactivo_id in reactivos_seleccionados:
            reactivo = self.algoritmo.reactivos_data[reactivo_id]
            todos_los_pesos.extend(reactivo.peso_habilidades.values())
        plt.hist(todos_los_pesos, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('Distribución de Pesos de Habilidades', fontweight='bold')
        
        plt.subplot(2, 2, 3)
        todas_habilidades = set()
        for reactivo_id in reactivos_seleccionados:
            reactivo = self.algoritmo.reactivos_data[reactivo_id]
            todas_habilidades.update(reactivo.habilidades)
        
        todas_habilidades = sorted(list(todas_habilidades))
        matriz_habilidades = np.zeros((len(reactivos_seleccionados), len(todas_habilidades)))
        
        for i, reactivo_id in enumerate(reactivos_seleccionados):
            reactivo = self.algoritmo.reactivos_data[reactivo_id]
            for j, habilidad_id in enumerate(todas_habilidades):
                if habilidad_id in reactivo.peso_habilidades:
                    matriz_habilidades[i, j] = reactivo.peso_habilidades[habilidad_id]
        
        sns.heatmap(matriz_habilidades, 
                   xticklabels=todas_habilidades, 
                   yticklabels=nombres_reactivos,
                   annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Mapa de Calor: Reactivos vs Habilidades', fontweight='bold')
        
        plt.subplot(2, 2, 4)
        metricas = mejor.metricas
        colores_metricas = ['green', 'red', 'orange', 'blue']
        bars = plt.bar(metricas.keys(), metricas.values(), color=colores_metricas, alpha=0.7)
        plt.title('Métricas del Mejor Individuo', fontweight='bold')
        
        for bar, valor in zip(bars, metricas.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def graficar_prediccion_post_reactivo(self, reactivo_id: str = None):
        """
        Gráficas de predicción después de ejecutar el reactivo con 3 salidas:
        1. Predicción de mejora de habilidades
        2. Probabilidad de aprobación futura
        3. Impacto en el rendimiento general
        """
        if reactivo_id is None and self.algoritmo.mejor_individuo:
            reactivo_id = self.algoritmo.mejor_individuo.genes[0]
        elif reactivo_id is None:
            print("No hay reactivo especificado ni mejor individuo disponible")
            return
            
        reactivo = self.algoritmo.reactivos_data.get(reactivo_id)
        if not reactivo:
            print(f"Reactivo {reactivo_id} no encontrado")
            return
            
        # Simulación de predicciones
        predicciones = self._calcular_predicciones_reactivo(reactivo_id)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Predicciones Post-Reactivo: {reactivo_id}', fontsize=16, fontweight='bold')
        
        # 1. PREDICCIÓN DE MEJORA DE HABILIDADES
        habilidades = list(predicciones['mejora_habilidades'].keys())
        mejoras = list(predicciones['mejora_habilidades'].values())
        
        bars1 = axes[0, 0].bar(habilidades, mejoras, 
                              color=['lightgreen' if m > 0 else 'lightcoral' for m in mejoras],
                              alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('1. Predicción de Mejora por Habilidad', fontweight='bold')
        axes[0, 0].set_ylabel('Mejora Esperada (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Agregar valores en las barras
        for bar, valor in zip(bars1, mejoras):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, height + (0.5 if height >= 0 else -1.5),
                           f'{valor:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                           fontweight='bold', fontsize=9)
        
        # 2. PROBABILIDAD DE APROBACIÓN FUTURA
        probabilidades = predicciones['probabilidad_aprobacion']
        habilidades_prob = list(probabilidades.keys())
        prob_valores = list(probabilidades.values())
        
        # Crear gráfico de radar/polar
        angles = np.linspace(0, 2 * np.pi, len(habilidades_prob), endpoint=False).tolist()
        prob_valores += prob_valores[:1]  # Cerrar el círculo
        angles += angles[:1]
        
        axes[0, 1].remove()
        ax_polar = fig.add_subplot(2, 2, 2, projection='polar')
        ax_polar.plot(angles, prob_valores, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax_polar.fill(angles, prob_valores, alpha=0.25, color='blue')
        ax_polar.set_xticks(angles[:-1])
        ax_polar.set_xticklabels(habilidades_prob, fontsize=9)
        ax_polar.set_ylim(0, 100)
        ax_polar.set_title('2. Probabilidad de Aprobación Futura (%)', fontweight='bold', pad=20)
        ax_polar.grid(True)
        
        # 3. IMPACTO EN RENDIMIENTO GENERAL
        tiempo = predicciones['timeline']
        rendimiento_actual = predicciones['rendimiento_actual']
        rendimiento_predicho = predicciones['rendimiento_predicho']
        
        axes[1, 0].plot(tiempo, rendimiento_actual, 'o-', label='Rendimiento Actual', 
                       linewidth=2, markersize=6, color='red')
        axes[1, 0].plot(tiempo, rendimiento_predicho, 's-', label='Predicción Post-Reactivo', 
                       linewidth=2, markersize=6, color='green')
        axes[1, 0].fill_between(tiempo, rendimiento_actual, rendimiento_predicho, 
                               alpha=0.3, color='lightgreen')
        axes[1, 0].set_title('3. Impacto en Rendimiento General', fontweight='bold')
        axes[1, 0].set_xlabel('Tiempo (semanas)')
        axes[1, 0].set_ylabel('Rendimiento General (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. RESUMEN DE MÉTRICAS CLAVE
        metricas_clave = {
            'Mejora Promedio': f"{np.mean(mejoras):.1f}%",
            'Probabilidad Promedio': f"{np.mean(prob_valores[:-1]):.1f}%",
            'Incremento Rendimiento': f"{rendimiento_predicho[-1] - rendimiento_actual[-1]:.1f}%",
            'Tiempo Estimado': f"{tiempo[-1]} semanas",
            'Habilidades Impactadas': len(habilidades),
            'Confianza Predicción': f"{predicciones['confianza']:.1f}%"
        }
        
        axes[1, 1].axis('off')
        y_pos = 0.9
        axes[1, 1].text(0.5, 0.95, 'RESUMEN DE PREDICCIONES', 
                       ha='center', va='top', fontsize=14, fontweight='bold',
                       transform=axes[1, 1].transAxes)
        
        for metrica, valor in metricas_clave.items():
            color = 'green' if any(word in metrica for word in ['Mejora', 'Incremento', 'Probabilidad']) else 'blue'
            axes[1, 1].text(0.1, y_pos, f'• {metrica}:', ha='left', va='top', 
                           fontweight='bold', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.7, y_pos, valor, ha='left', va='top', 
                           color=color, fontweight='bold', transform=axes[1, 1].transAxes)
            y_pos -= 0.12
        
        plt.tight_layout()
        plt.show()
    
    def graficar_prediccion_completitud_arbol(self):
        """
        Gráficas de predicción de completitud de habilidades tipo árbol
        después de aplicar cada reactivo.
        """
        if not self.algoritmo.mejor_individuo:
            print("No hay mejor individuo disponible")
            return
            
        reactivos_seleccionados = self.algoritmo.mejor_individuo.genes
        predicciones_arbol = self._calcular_predicciones_arbol(reactivos_seleccionados)
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
        
        # 1. ÁRBOL PRINCIPAL DE DEPENDENCIAS
        ax_arbol = fig.add_subplot(gs[0, :2])
        self._dibujar_arbol_habilidades(ax_arbol, predicciones_arbol)
        
        # 2. PROGRESIÓN TEMPORAL
        ax_progresion = fig.add_subplot(gs[0, 2])
        self._dibujar_progresion_temporal(ax_progresion, predicciones_arbol)
        
        # 3. MÉTRICAS DE COMPLETITUD POR REACTIVO
        ax_metricas = fig.add_subplot(gs[1, :])
        self._dibujar_metricas_completitud(ax_metricas, predicciones_arbol, reactivos_seleccionados)
        
        # 4. ANÁLISIS DE RIESGO Y OPORTUNIDADES
        ax_riesgo = fig.add_subplot(gs[2, :])
        self._dibujar_analisis_riesgo(ax_riesgo, predicciones_arbol)
        
        plt.suptitle('Predicción de Completitud de Habilidades - Vista de Árbol', 
                     fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _calcular_predicciones_reactivo(self, reactivo_id: str) -> Dict:
        """Calcula las predicciones específicas para un reactivo."""
        reactivo = self.algoritmo.reactivos_data[reactivo_id]
        
        # Simular mejoras en habilidades
        mejora_habilidades = {}
        probabilidad_aprobacion = {}
        
        for habilidad_id in reactivo.habilidades:
            habilidad_actual = self.algoritmo.habilidades_data[habilidad_id]
            peso = reactivo.peso_habilidades.get(habilidad_id, 0.1)
            
            # Calcular mejora esperada basada en el peso y estado actual
            mejora_base = peso * 15  # Factor de mejora base
            mejora_actual = mejora_base * (1 - habilidad_actual.calificacion)  # Más mejora si está más bajo
            mejora_habilidades[habilidad_id] = mejora_actual
            
            # Calcular probabilidad de aprobación
            prob_base = min(95, (habilidad_actual.calificacion + mejora_actual/100) * 100)
            probabilidad_aprobacion[habilidad_id] = prob_base
        
        # Simular evolución temporal del rendimiento
        timeline = list(range(0, 13, 2))  # 0 a 12 semanas, cada 2 semanas
        rendimiento_actual = [60, 61, 62, 63, 64, 65, 66]  # Tendencia gradual
        
        # Predicción con reactivo aplicado
        boost_inicial = np.mean(list(mejora_habilidades.values())) * 0.5
        rendimiento_predicho = [
            60, 62 + boost_inicial*0.3, 65 + boost_inicial*0.6, 
            69 + boost_inicial*0.8, 72 + boost_inicial, 
            75 + boost_inicial*1.1, 78 + boost_inicial*1.2
        ]
        
        return {
            'mejora_habilidades': mejora_habilidades,
            'probabilidad_aprobacion': probabilidad_aprobacion,
            'timeline': timeline,
            'rendimiento_actual': rendimiento_actual,
            'rendimiento_predicho': rendimiento_predicho,
            'confianza': min(95, 70 + len(reactivo.habilidades) * 5)
        }
    
    def _calcular_predicciones_arbol(self, reactivos: List[str]) -> Dict:
        """Calcula predicciones para la estructura de árbol de habilidades."""
        todas_habilidades = set()
        for reactivo_id in reactivos:
            todas_habilidades.update(self.algoritmo.reactivos_data[reactivo_id].habilidades)
        
        todas_habilidades = sorted(list(todas_habilidades))
        
        # Simular dependencias entre habilidades (árbol)
        dependencias = self._generar_dependencias_habilidades(todas_habilidades)
        
        # Calcular completitud progresiva
        completitud_por_etapa = []
        for i, reactivo_id in enumerate(reactivos):
            reactivo = self.algoritmo.reactivos_data[reactivo_id]
            completitud_etapa = {}
            
            for habilidad_id in todas_habilidades:
                # Calcular completitud basada en reactivos anteriores y actuales
                completitud_base = self.algoritmo.habilidades_data[habilidad_id].calificacion
                
                if habilidad_id in reactivo.habilidades:
                    peso = reactivo.peso_habilidades[habilidad_id]
                    incremento = peso * 0.3  # Factor de incremento
                    completitud_base = min(1.0, completitud_base + incremento)
                
                completitud_etapa[habilidad_id] = completitud_base
            
            completitud_por_etapa.append(completitud_etapa)
        
        return {
            'habilidades': todas_habilidades,
            'dependencias': dependencias,
            'completitud_por_etapa': completitud_por_etapa,
            'reactivos': reactivos
        }
    
    def _generar_dependencias_habilidades(self, habilidades: List[str]) -> Dict:
        """Genera un grafo de dependencias simulado entre habilidades."""
        dependencias = {}
        
        # Simular algunas dependencias lógicas
        for i, habilidad in enumerate(habilidades):
            dependencias[habilidad] = []
            
            # Agregar dependencias simuladas (las primeras habilidades dependen de las anteriores)
            if i > 0:
                # Cada habilidad puede depender de 1-2 habilidades anteriores
                num_deps = min(2, i)
                deps_indices = np.random.choice(i, size=num_deps, replace=False)
                dependencias[habilidad] = [habilidades[idx] for idx in deps_indices]
        
        return dependencias
    
    def _dibujar_arbol_habilidades(self, ax, predicciones_arbol):
        """Dibuja el árbol de dependencias de habilidades."""
        G = nx.DiGraph()
        
        # Agregar nodos y aristas
        for habilidad in predicciones_arbol['habilidades']:
            G.add_node(habilidad)
            
        for habilidad, deps in predicciones_arbol['dependencias'].items():
            for dep in deps:
                G.add_edge(dep, habilidad)
        
        # Layout del grafo
        try:
            pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Obtener completitud final
        completitud_final = predicciones_arbol['completitud_por_etapa'][-1]
        
        # Dibujar nodos con colores según completitud
        node_colors = [completitud_final[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              cmap='RdYlGn', vmin=0, vmax=1, 
                              node_size=1000, ax=ax, alpha=0.8)
        
        # Dibujar aristas
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, 
                              alpha=0.6, ax=ax)
        
        # Etiquetas
        nx.draw_networkx_labels(G, pos, font_size=8, 
                               font_weight='bold', ax=ax)
        
        ax.set_title('Árbol de Dependencias de Habilidades', fontweight='bold')
        ax.axis('off')
        
        # Agregar colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Completitud', rotation=270, labelpad=15)
    
    def _dibujar_progresion_temporal(self, ax, predicciones_arbol):
        """Dibuja la progresión temporal de completitud."""
        etapas = len(predicciones_arbol['completitud_por_etapa'])
        habilidades = predicciones_arbol['habilidades']
        
        for habilidad in habilidades[:5]:  # Mostrar solo las primeras 5 para claridad
            completitudes = [etapa[habilidad] for etapa in predicciones_arbol['completitud_por_etapa']]
            ax.plot(range(etapas), completitudes, 'o-', label=habilidad, linewidth=2, markersize=6)
        
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Umbral Aprobación')
        ax.set_xlabel('Etapa de Reactivo')
        ax.set_ylabel('Completitud')
        ax.set_title('Progresión Temporal\nde Habilidades', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _dibujar_metricas_completitud(self, ax, predicciones_arbol, reactivos):
        """Dibuja métricas de completitud por reactivo."""
        etapas = len(predicciones_arbol['completitud_por_etapa'])
        
        completitud_promedio = []
        habilidades_aprobadas = []
        
        for etapa in predicciones_arbol['completitud_por_etapa']:
            promedio = np.mean(list(etapa.values()))
            aprobadas = sum(1 for v in etapa.values() if v >= 0.7)
            completitud_promedio.append(promedio * 100)
            habilidades_aprobadas.append(aprobadas)
        
        # Gráfico de barras doble
        x = np.arange(etapas)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, completitud_promedio, width, 
                      label='Completitud Promedio (%)', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, habilidades_aprobadas, width, 
                      label='Habilidades Aprobadas', color='lightgreen', alpha=0.8)
        
        ax.set_xlabel('Etapa de Reactivo')
        ax.set_ylabel('Valor')
        ax.set_title('Métricas de Completitud por Reactivo', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'R{i+1}' for i in range(etapas)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    def _dibujar_analisis_riesgo(self, ax, predicciones_arbol):
        """Dibuja análisis de riesgo y oportunidades."""
        habilidades = predicciones_arbol['habilidades']
        completitud_final = predicciones_arbol['completitud_por_etapa'][-1]
        
        # Categorizar habilidades por riesgo
        alto_riesgo = [h for h in habilidades if completitud_final[h] < 0.5]
        medio_riesgo = [h for h in habilidades if 0.5 <= completitud_final[h] < 0.7]
        bajo_riesgo = [h for h in habilidades if completitud_final[h] >= 0.7]
        
        categorias = ['Alto Riesgo\n(< 50%)', 'Medio Riesgo\n(50-70%)', 'Bajo Riesgo\n(≥ 70%)']
        cantidades = [len(alto_riesgo), len(medio_riesgo), len(bajo_riesgo)]
        colores = ['red', 'orange', 'green']
        
        bars = ax.bar(categorias, cantidades, color=colores, alpha=0.7, edgecolor='black')
        
        ax.set_title('Análisis de Riesgo - Distribución de Habilidades', fontweight='bold')
        ax.set_ylabel('Número de Habilidades')
        
        # Agregar valores y porcentajes
        total = sum(cantidades)
        for bar, cantidad in zip(bars, cantidades):
            porcentaje = (cantidad / total) * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{cantidad}\n({porcentaje:.1f}%)', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar texto de recomendaciones
        recomendaciones = f"""
        Recomendaciones:
        • {len(alto_riesgo)} habilidades requieren atención inmediata
        • Enfocar próximos reactivos en habilidades de alto riesgo
        • {len(bajo_riesgo)} habilidades están en buen camino
        """
        
        ax.text(1.02, 0.5, recomendaciones, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor="lightblue", alpha=0.5))
    
    def generar_predicciones_completas(self):
        """Genera todas las gráficas de predicción."""
        print("Generando gráficas de predicción...")
        
        if self.algoritmo.mejor_individuo:
            # Gráficas de predicción post-reactivo para el primer reactivo seleccionado
            primer_reactivo = self.algoritmo.mejor_individuo.genes[0]
            print(f"Generando predicciones para reactivo: {primer_reactivo}")
            self.graficar_prediccion_post_reactivo(primer_reactivo)
            
            # Gráficas de completitud tipo árbol
            print("Generando predicciones de completitud tipo árbol...")
            self.graficar_prediccion_completitud_arbol()
        else:
            print("No hay mejor individuo disponible para generar predicciones")
    
    def generar_reporte_completo(self):
        """Genera un reporte completo con todas las visualizaciones."""
        print("="*60)
        print("REPORTE COMPLETO DEL ALGORITMO GENÉTICO")
        print("="*60)
        
        if self.algoritmo.mejor_individuo:
            mejor = self.algoritmo.mejor_individuo
            print(f"\nMEJOR SOLUCIÓN ENCONTRADA:")
            print(f"Reactivos seleccionados: {mejor.genes}")
            print(f"Fitness: {mejor.fitness:.4f}")
            print(f"Métricas:")
            for metrica, valor in mejor.metricas.items():
                print(f"  {metrica}: {valor:.4f}")
            
            simulacion = mejor.simular_mejora()
            print(f"\nSIMULACIÓN DE MEJORA:")
            print(f"Fitness actual: {simulacion['fitness_antes']:.4f}")
            print(f"Fitness esperado: {simulacion['fitness_despues']:.4f}")
            print(f"Mejora esperada: {simulacion['mejora_fitness']:.4f}")
            print(f"Habilidades aprobadas antes: {simulacion['habilidades_aprobadas_antes']}")
            print(f"Habilidades aprobadas después: {simulacion['habilidades_aprobadas_despues']}")
            print(f"Nuevas habilidades que se aprobarían: {simulacion['nuevas_habilidades_aprobadas']}")
        
        print("\nGenerando visualizaciones...")
        self.graficar_evolucion_fitness()
        self.graficar_metricas_objetivos()
        self.graficar_estado_habilidades()
        self.graficar_analisis_reactivos_seleccionados()
        
        # Nuevas visualizaciones de predicción
        print("\nGenerando predicciones...")
        self.generar_predicciones_completas()