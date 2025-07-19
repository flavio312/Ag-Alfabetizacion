import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Optional
from models.individuo import Individuo

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