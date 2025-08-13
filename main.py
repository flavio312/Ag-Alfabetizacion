import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import networkx as nx

@dataclass
class Reactivo:
    """Representa un reactivo educativo con sus habilidades asociadas."""
    id: str
    habilidades: List[str]
    peso_habilidades: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.peso_habilidades:
            # Si no se especifican pesos, distribuir equitativamente
            peso_igual = 1.0 / len(self.habilidades)
            self.peso_habilidades = {h: peso_igual for h in self.habilidades}
        else:
            # Normalizar pesos existentes para que sumen 1
            suma_pesos = sum(self.peso_habilidades.values())
            if suma_pesos != 1.0:
                self.peso_habilidades = {h: peso/suma_pesos for h, peso in self.peso_habilidades.items()}

@dataclass
class Habilidad:
    """Representa una habilidad con su calificación actual."""
    id: str
    calificacion: float = 0.0
    aprobada: bool = False
    umbral_aprobacion: float = 0.7
    
    def __post_init__(self):
        self.aprobada = self.calificacion >= self.umbral_aprobacion

class Individual:
    """Representa un individuo en el algoritmo genético."""
    
    def __init__(self, genes: List[str], K: int, reactivos_data: Dict[str, Reactivo], 
                 habilidades_data: Dict[str, Habilidad], conteo_reactivos: Dict[str, int]):
        self.genes = genes[:K]  # Solo los primeros K reactivos
        self.K = K
        self.reactivos_data = reactivos_data
        self.habilidades_data = habilidades_data
        self.conteo_reactivos = conteo_reactivos
        self.fitness = 0.0
        self.metricas = {}
        self._calcular_fitness()
    
    def _calcular_obj1(self) -> float:
        """OBJ1: Maximizar uso de reactivos con habilidades no aprobadas."""
        total_habilidades_no_aprobadas = 0
        total_habilidades = 0
        
        for reactivo_id in self.genes:
            reactivo = self.reactivos_data[reactivo_id]
            for habilidad_id, peso in reactivo.peso_habilidades.items():
                total_habilidades += peso
                if not self.habilidades_data[habilidad_id].aprobada:
                    # Priorizar habilidades más bajas
                    calificacion = self.habilidades_data[habilidad_id].calificacion
                    factor_prioridad = (1.0 - calificacion) if calificacion > 0 else 1.0
                    total_habilidades_no_aprobadas += peso * factor_prioridad
        
        return total_habilidades_no_aprobadas / max(total_habilidades, 1)
    
    def _calcular_obj2(self) -> float:
        """OBJ2: Minimizar uso de reactivos ya realizados."""
        return sum(self.conteo_reactivos.get(reactivo_id, 0) for reactivo_id in self.genes)
    
    def _calcular_obj3(self) -> float:
        """OBJ3: Minimizar uso de reactivos con habilidades ya aprobadas."""
        reactivos_con_habilidades_aprobadas = 0
        
        for reactivo_id in self.genes:
            reactivo = self.reactivos_data[reactivo_id]
            tiene_habilidad_aprobada = any(
                self.habilidades_data[h].aprobada for h in reactivo.habilidades
            )
            if tiene_habilidad_aprobada:
                reactivos_con_habilidades_aprobadas += 1
        
        return reactivos_con_habilidades_aprobadas
    
    def _calcular_obj4(self) -> float:
        """OBJ4: Maximizar cantidad de habilidades involucradas."""
        habilidades_involucradas = set()
        
        for reactivo_id in self.genes:
            reactivo = self.reactivos_data[reactivo_id]
            habilidades_involucradas.update(reactivo.habilidades)
        
        return len(habilidades_involucradas)
    
    def _calcular_fitness(self):
        """Calcula el fitness según la fórmula especificada."""
        obj1 = self._calcular_obj1()
        obj2 = self._calcular_obj2()
        obj3 = self._calcular_obj3()
        obj4 = self._calcular_obj4()
        
        self.metricas = {
            'OBJ1': obj1,
            'OBJ2': obj2,
            'OBJ3': obj3,
            'OBJ4': obj4
        }
        
        numerador = (1 + obj1) * (1 + obj4)
        denominador = (1 + obj2) * (1 + obj3)
        self.fitness = numerador / denominador
    
    def copy(self):
        """Crea una copia del individuo."""
        return Individual(
            genes=self.genes.copy(),
            K=self.K,
            reactivos_data=self.reactivos_data,
            habilidades_data=self.habilidades_data,
            conteo_reactivos=self.conteo_reactivos
        )

class AlgoritmoGenetico:
    """Implementación del algoritmo genético para selección de reactivos."""
    def __init__(self, reactivos_alcanzables: List[str], reactivos_data: Dict[str, Reactivo],
                 habilidades_data: Dict[str, Habilidad], conteo_reactivos: Dict[str, int],
                 K: int = 3, tamaño_poblacion: int = 20, generaciones: int = 50,
                 tasa_mutacion: float = 0.1, presion_seleccion: float = 0.7):
        
        self.reactivos_alcanzables = reactivos_alcanzables
        self.reactivos_data = reactivos_data
        self.habilidades_data = habilidades_data
        self.conteo_reactivos = conteo_reactivos
        self.K = K
        self.tamaño_poblacion = tamaño_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion
        self.presion_seleccion = presion_seleccion
        
        self.poblacion: List[Individual] = []
        self.mejor_individuo: Optional[Individual] = None
        self.historial_fitness: List[float] = []
        self.historial_metricas: List[Dict] = []
        
    def inicializar_poblacion(self):
        """Crea la población inicial de manera diversa."""
        self.poblacion = []
        
        for _ in range(self.tamaño_poblacion):
            genes = random.sample(self.reactivos_alcanzables, 
                                min(self.K, len(self.reactivos_alcanzables)))
            # Completar con reactivos aleatorios si es necesario
            while len(genes) < self.K:
                reactivo_extra = random.choice(self.reactivos_alcanzables)
                if reactivo_extra not in genes:
                    genes.append(reactivo_extra)
            
            individuo = Individual(genes, self.K, self.reactivos_data, 
                                 self.habilidades_data, self.conteo_reactivos)
            self.poblacion.append(individuo)
    
    def seleccionar_padres(self) -> Tuple[Individual, Individual]:
        """Selección por torneo."""
        tamaño_torneo = max(2, int(len(self.poblacion) * self.presion_seleccion))
        
        # Seleccionar padre 1
        candidatos1 = random.sample(self.poblacion, min(tamaño_torneo, len(self.poblacion)))
        padre1 = max(candidatos1, key=lambda x: x.fitness)
        
        # Seleccionar padre 2 (diferente al padre 1)
        candidatos2 = [ind for ind in self.poblacion if ind != padre1]
        if candidatos2:
            candidatos2 = random.sample(candidatos2, min(tamaño_torneo, len(candidatos2)))
            padre2 = max(candidatos2, key=lambda x: x.fitness)
        else:
            padre2 = padre1
        
        return padre1, padre2
    
    def cruzar(self, padre1: Individual, padre2: Individual) -> Tuple[Individual, Individual]:
        """Cruza con corrección de duplicados."""
        punto_cruza = random.randint(1, len(padre1.genes) - 1)
        
        # Crear hijos iniciales
        hijo1_genes = padre1.genes[:punto_cruza] + padre2.genes[punto_cruza:]
        hijo2_genes = padre2.genes[:punto_cruza] + padre1.genes[punto_cruza:]
        
        # Corregir duplicados
        hijo1_genes = self._corregir_duplicados(hijo1_genes, padre1.genes + padre2.genes)
        hijo2_genes = self._corregir_duplicados(hijo2_genes, padre1.genes + padre2.genes)
        
        hijo1 = Individual(hijo1_genes, self.K, self.reactivos_data, 
                          self.habilidades_data, self.conteo_reactivos)
        hijo2 = Individual(hijo2_genes, self.K, self.reactivos_data, 
                          self.habilidades_data, self.conteo_reactivos)
        
        return hijo1, hijo2
    
    def _corregir_duplicados(self, genes: List[str], genes_padres: List[str]) -> List[str]:
        """Corrige duplicados en los genes según la estrategia especificada."""
        genes_corregidos = genes.copy()
        todos_reactivos = set(self.reactivos_alcanzables)
        reactivos_en_genes = set(genes_corregidos)
        reactivos_faltantes = list(todos_reactivos - reactivos_en_genes)
        
        # Buscar duplicados
        i = 0
        while i < len(genes_corregidos) and reactivos_faltantes:
            reactivo_actual = genes_corregidos[i]
            # Contar ocurrencias del reactivo actual desde la posición i hacia adelante
            ocurrencias = [j for j in range(i+1, len(genes_corregidos)) 
                          if genes_corregidos[j] == reactivo_actual]
            
            if ocurrencias:
                # Reemplazar una ocurrencia duplicada
                pos_duplicado = random.choice(ocurrencias)
                reactivo_reemplazo = random.choice(reactivos_faltantes)
                genes_corregidos[pos_duplicado] = reactivo_reemplazo
                reactivos_faltantes.remove(reactivo_reemplazo)
            
            i += 1
        
        return genes_corregidos
    
    def mutar(self, individuo: Individual) -> Individual:
        """Aplica mutación a un individuo."""
        if random.random() < self.tasa_mutacion:
            individuo_mutado = individuo.copy()
            pos = random.randint(0, len(individuo_mutado.genes) - 1)
            
            reactivos_disponibles = [r for r in self.reactivos_alcanzables 
                                   if r not in individuo_mutado.genes]
            
            if reactivos_disponibles:
                nuevo_reactivo = random.choice(reactivos_disponibles)
                individuo_mutado.genes[pos] = nuevo_reactivo
                individuo_mutado._calcular_fitness()
            
            return individuo_mutado
        
        return individuo
    
    def seleccionar_supervivientes(self, poblacion_expandida: List[Individual]):
        """Selecciona supervivientes para la siguiente generación."""
        # Ordenar por fitness descendente
        poblacion_ordenada = sorted(poblacion_expandida, key=lambda x: x.fitness, reverse=True)
        
        # Mantener élite (50%) y algunos aleatorios para diversidad (25%)
        tamaño_elite = self.tamaño_poblacion // 2
        elite = poblacion_ordenada[:tamaño_elite]
        
        tamaño_aleatorio = self.tamaño_poblacion // 4
        resto = poblacion_ordenada[tamaño_elite:]
        
        if resto and tamaño_aleatorio > 0:
            aleatorios = random.sample(resto, min(tamaño_aleatorio, len(resto)))
        else:
            aleatorios = []
        
        self.poblacion = elite + aleatorios
        
        # Completar población si es necesario
        while len(self.poblacion) < self.tamaño_poblacion:
            self.poblacion.append(random.choice(poblacion_ordenada[:self.tamaño_poblacion]))
    
    def evolucionar(self):
        """Ejecuta el algoritmo genético."""
        self.inicializar_poblacion()
        
        for generacion in range(self.generaciones):
            # Guardar estadísticas
            mejor_actual = max(self.poblacion, key=lambda x: x.fitness)
            self.historial_fitness.append(mejor_actual.fitness)
            self.historial_metricas.append(mejor_actual.metricas.copy())
            
            if self.mejor_individuo is None or mejor_actual.fitness > self.mejor_individuo.fitness:
                self.mejor_individuo = mejor_actual.copy()
            
            # Crear nueva generación
            nueva_poblacion = []
            
            # Generar descendencia
            while len(nueva_poblacion) < self.tamaño_poblacion:
                padre1, padre2 = self.seleccionar_padres()
                hijo1, hijo2 = self.cruzar(padre1, padre2)
                
                # Aplicar mutación
                hijo1 = self.mutar(hijo1)
                hijo2 = self.mutar(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
            
            # Combinar población actual con descendencia
            poblacion_expandida = self.poblacion + nueva_poblacion
            
            # Seleccionar supervivientes
            self.seleccionar_supervivientes(poblacion_expandida)
            
            if generacion % 10 == 0:
                print(f"Generación {generacion}: Mejor fitness = {mejor_actual.fitness:.4f}")
    
    def simular_mejora(self, individuo: Individual) -> Dict:
        """Simula la mejora esperada si se resuelven correctamente los reactivos (método original)."""
        habilidades_temp = deepcopy(self.habilidades_data)
        
        # Simular que se resuelven correctamente los reactivos
        for reactivo_id in individuo.genes:
            reactivo = self.reactivos_data[reactivo_id]
            for habilidad_id, peso in reactivo.peso_habilidades.items():
                habilidad = habilidades_temp[habilidad_id]
                if habilidad.calificacion < 0.7:
                    # Mejorar la habilidad proporcionalmente al peso
                    mejora = peso * 0.3  # Mejora base
                    nueva_calificacion = min(1.0, habilidad.calificacion + mejora)
                    habilidad.calificacion = nueva_calificacion
                    habilidad.aprobada = nueva_calificacion >= habilidad.umbral_aprobacion
        
        # Crear individuo temporal con habilidades mejoradas
        individuo_temp = Individual(
            genes=individuo.genes.copy(),
            K=individuo.K,
            reactivos_data=self.reactivos_data,
            habilidades_data=habilidades_temp,
            conteo_reactivos=self.conteo_reactivos
        )
        
        # Calcular métricas de mejora
        habilidades_aprobadas_antes = sum(1 for h in self.habilidades_data.values() if h.aprobada)
        habilidades_aprobadas_despues = sum(1 for h in habilidades_temp.values() if h.aprobada)
        
        return {
            'fitness_antes': individuo.fitness,
            'fitness_despues': individuo_temp.fitness,
            'mejora_fitness': individuo_temp.fitness - individuo.fitness,
            'habilidades_aprobadas_antes': habilidades_aprobadas_antes,
            'habilidades_aprobadas_despues': habilidades_aprobadas_despues,
            'nuevas_habilidades_aprobadas': habilidades_aprobadas_despues - habilidades_aprobadas_antes
        }
    
    def simular_mejora_secuencial(self, individuo: Individual) -> List[Dict]:
        """Simula la mejora secuencial aplicando cada reactivo paso a paso."""
        resultados = []
        habilidades_temp = deepcopy(self.habilidades_data)
        
        # Estado inicial
        estado_inicial = {
            'paso': 0,
            'reactivo': 'Estado Inicial',
            'habilidades': {h_id: h.calificacion for h_id, h in habilidades_temp.items()},
            'habilidades_aprobadas': sum(1 for h in habilidades_temp.values() if h.aprobada),
            'completitud': sum(1 for h in habilidades_temp.values() if h.aprobada) / len(habilidades_temp),
            'mejora_promedio': 0.0
        }
        resultados.append(estado_inicial)
        
        # Aplicar cada reactivo secuencialmente
        for i, reactivo_id in enumerate(individuo.genes):
            reactivo = self.reactivos_data[reactivo_id]
            mejoras_aplicadas = {}
            
            # Calcular mejoras para cada habilidad del reactivo
            for habilidad_id, peso in reactivo.peso_habilidades.items():
                habilidad = habilidades_temp[habilidad_id]
                calificacion_anterior = habilidad.calificacion
                
                if habilidad.calificacion < 0.7:
                    # Aplicar mejora basada en el peso del reactivo
                    mejora_base = peso * 0.25  # Factor de mejora ajustable
                    # Agregar algo de variabilidad realista
                    variabilidad = random.uniform(-0.05, 0.1)
                    mejora_total = mejora_base + variabilidad
                    
                    nueva_calificacion = min(1.0, habilidad.calificacion + mejora_total)
                    habilidad.calificacion = nueva_calificacion
                    habilidad.aprobada = nueva_calificacion >= habilidad.umbral_aprobacion
                    
                    mejoras_aplicadas[habilidad_id] = nueva_calificacion - calificacion_anterior
            
            # Registrar estado después de aplicar el reactivo
            estado_paso = {
                'paso': i + 1,
                'reactivo': reactivo_id,
                'habilidades': {h_id: h.calificacion for h_id, h in habilidades_temp.items()},
                'habilidades_aprobadas': sum(1 for h in habilidades_temp.values() if h.aprobada),
                'completitud': sum(1 for h in habilidades_temp.values() if h.aprobada) / len(habilidades_temp),
                'mejoras_aplicadas': mejoras_aplicadas,
                'mejora_promedio': np.mean(list(mejoras_aplicadas.values())) if mejoras_aplicadas else 0.0,
                'habilidades_involucradas': list(reactivo.habilidades)
            }
            resultados.append(estado_paso)
        
        return resultados

class VisualizadorResultados:
    """Clase para generar visualizaciones del algoritmo genético."""
    
    def __init__(self, algoritmo: AlgoritmoGenetico):
        self.algoritmo = algoritmo
        plt.style.use('seaborn-v0_8')
    
    def graficar_evolucion_fitness(self):
        """Gráfica la evolución del fitness a lo largo de las generaciones."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.algoritmo.historial_fitness, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title('Evolución del Fitness', fontsize=14, fontweight='bold')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Gráfica de las últimas 10 generaciones con más detalle
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
        
        # OBJ1: Maximizar (mejor cuando es alto)
        axes[0, 0].plot(metricas_df['OBJ1'], 'g-', linewidth=2, marker='o')
        axes[0, 0].set_title('OBJ1: Habilidades No Aprobadas', fontweight='bold')
        axes[0, 0].set_xlabel('Generación')
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # OBJ2: Minimizar (mejor cuando es bajo)
        axes[0, 1].plot(metricas_df['OBJ2'], 'r-', linewidth=2, marker='s')
        axes[0, 1].set_title('OBJ2: Reactivos Ya Realizados', fontweight='bold')
        axes[0, 1].set_xlabel('Generación')
        axes[0, 1].set_ylabel('Conteo')
        axes[0, 1].grid(True, alpha=0.3)
        
        # OBJ3: Minimizar (mejor cuando es bajo)
        axes[1, 0].plot(metricas_df['OBJ3'], 'orange', linewidth=2, marker='^')
        axes[1, 0].set_title('OBJ3: Reactivos con Habilidades Aprobadas', fontweight='bold')
        axes[1, 0].set_xlabel('Generación')
        axes[1, 0].set_ylabel('Cantidad')
        axes[1, 0].grid(True, alpha=0.3)
        
        # OBJ4: Maximizar (mejor cuando es alto)
        axes[1, 1].plot(metricas_df['OBJ4'], 'purple', linewidth=2, marker='d')
        axes[1, 1].set_title('OBJ4: Habilidades Involucradas', fontweight='bold')
        axes[1, 1].set_xlabel('Generación')
        axes[1, 1].set_ylabel('Cantidad')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def graficar_estado_habilidades(self):
        """Gráfica el estado actual de las habilidades del estudiante."""
        habilidades_data = list(self.algoritmo.habilidades_data.items())
        nombres = [h[0] for h in habilidades_data]
        calificaciones = [h[1].calificacion for h in habilidades_data]
        aprobadas = [h[1].aprobada for h in habilidades_data]
        
        plt.figure(figsize=(14, 8))
        
        # Gráfica de barras de calificaciones
        plt.subplot(2, 1, 1)
        colores = ['green' if aprobada else 'red' for aprobada in aprobadas]
        bars = plt.bar(nombres, calificaciones, color=colores, alpha=0.7)
        plt.axhline(y=0.7, color='black', linestyle='--', alpha=0.8, label='Umbral de Aprobación')
        plt.title('Estado de Habilidades del Estudiante', fontsize=16, fontweight='bold')
        plt.ylabel('Calificación')
        plt.ylim(0, 1)
        plt.legend()
        plt.xticks(rotation=45)
        
        # Agregar valores sobre las barras
        for bar, calificacion in zip(bars, calificaciones):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{calificacion:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfica de pastel de habilidades aprobadas vs no aprobadas
        plt.subplot(2, 1, 2)
        aprobadas_count = sum(aprobadas)
        no_aprobadas_count = len(aprobadas) - aprobadas_count
        
        labels = ['Aprobadas', 'No Aprobadas']
        sizes = [aprobadas_count, no_aprobadas_count]
        colors = ['lightgreen', 'lightcoral']
        explode = (0.05, 0.05)
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=explode, shadow=True)
        plt.title('Distribución de Habilidades', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def graficar_analisis_reactivos_seleccionados(self):
        """Analiza y gráfica los reactivos seleccionados por el mejor individuo."""
        if not self.algoritmo.mejor_individuo:
            return
        
        mejor = self.algoritmo.mejor_individuo
        
        plt.figure(figsize=(16, 10))
        
        # Análisis de habilidades por reactivo
        plt.subplot(2, 2, 1)
        reactivos_seleccionados = mejor.genes
        habilidades_por_reactivo = []
        nombres_reactivos = []
        
        for reactivo_id in reactivos_seleccionados:
            reactivo = self.algoritmo.reactivos_data[reactivo_id]
            habilidades_por_reactivo.append(len(reactivo.habilidades))
            nombres_reactivos.append(reactivo_id)
        
        plt.bar(nombres_reactivos, habilidades_por_reactivo, color='skyblue', alpha=0.8)
        plt.title('Habilidades por Reactivo Seleccionado', fontweight='bold')
        plt.xlabel('Reactivo')
        plt.ylabel('Número de Habilidades')
        plt.xticks(rotation=45)
        
        # Distribución de pesos de habilidades
        plt.subplot(2, 2, 2)
        todos_los_pesos = []
        for reactivo_id in reactivos_seleccionados:
            reactivo = self.algoritmo.reactivos_data[reactivo_id]
            todos_los_pesos.extend(reactivo.peso_habilidades.values())
        
        plt.hist(todos_los_pesos, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('Distribución de Pesos de Habilidades', fontweight='bold')
        plt.xlabel('Peso')
        plt.ylabel('Frecuencia')
        
        # Mapa de calor de reactivos vs habilidades
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
        plt.xlabel('Habilidades')
        plt.ylabel('Reactivos')
        
        # Métricas del mejor individuo
        plt.subplot(2, 2, 4)
        metricas = mejor.metricas
        nombres_metricas = list(metricas.keys())
        valores_metricas = list(metricas.values())
        
        colores_metricas = ['green', 'red', 'orange', 'blue']
        bars = plt.bar(nombres_metricas, valores_metricas, color=colores_metricas, alpha=0.7)
        plt.title('Métricas del Mejor Individuo', fontweight='bold')
        plt.ylabel('Valor')
        
        # Agregar valores sobre las barras
        for bar, valor in zip(bars, valores_metricas):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def generar_reporte_completo(self):
        """Genera un reporte completo con todas las visualizaciones originales."""
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
            
            # Simulación de mejora
            simulacion = self.algoritmo.simular_mejora_secuencial(mejor)
            if simulacion:
                estado_final = simulacion[-1]
                estado_inicial = simulacion[0]
                print(f"\nSIMULACIÓN DE MEJORA:")
                print(f"Completitud inicial: {estado_inicial['completitud']*100:.1f}%")
                print(f"Completitud final esperada: {estado_final['completitud']*100:.1f}%")
                print(f"Habilidades aprobadas inicial: {estado_inicial['habilidades_aprobadas']}")
                print(f"Habilidades aprobadas final: {estado_final['habilidades_aprobadas']}")
                print(f"Nuevas habilidades que se aprobarían: {estado_final['habilidades_aprobadas'] - estado_inicial['habilidades_aprobadas']}")
        
        print("\nGenerando visualizaciones originales...")
        self.graficar_evolucion_fitness()
        self.graficar_metricas_objetivos()
        self.graficar_estado_habilidades()
        self.graficar_analisis_reactivos_seleccionados()

    # NUEVAS FUNCIONES DE PREDICCIÓN
    def graficar_prediccion_secuencial(self):
        """Genera las gráficas de predicción secuencial solicitadas."""
        if not self.algoritmo.mejor_individuo:
            print("No hay mejor individuo para analizar")
            return
        
        # Obtener simulación secuencial
        simulacion = self.algoritmo.simular_mejora_secuencial(self.algoritmo.mejor_individuo)
        
        # Crear figura con 3 subplots como solicitado
        fig = plt.figure(figsize=(20, 15))
        
        # GRÁFICA 1: Evolución de completitud por paso (tipo árbol/flujo)
        ax1 = plt.subplot(3, 1, 1)
        self._graficar_arbol_completitud(simulacion, ax1)
        
        # GRÁFICA 2: Matriz de calor de evolución de habilidades
        ax2 = plt.subplot(3, 1, 2)
        self._graficar_matriz_evolucion_habilidades(simulacion, ax2)
        
        # GRÁFICA 3: Métricas de impacto por reactivo
        ax3 = plt.subplot(3, 1, 3)
        self._graficar_impacto_reactivos(simulacion, ax3)
        
        plt.tight_layout()
        plt.show()
    
    def _graficar_arbol_completitud(self, simulacion: List[Dict], ax):
        """Gráfica tipo árbol mostrando la evolución de completitud."""
        ax.clear()
        
        # Extraer datos
        pasos = [s['paso'] for s in simulacion]
        completitudes = [s['completitud'] * 100 for s in simulacion]
        reactivos = [s['reactivo'] for s in simulacion]
        
        # Crear gráfica de flujo tipo árbol
        for i in range(len(pasos)):
            # Nodo principal
            color = 'lightgreen' if completitudes[i] >= 70 else 'lightcoral' if completitudes[i] >= 50 else 'lightblue'
            
            # Dibujar nodo
            circle = plt.Circle((i, completitudes[i]), 15, color=color, alpha=0.7, zorder=3)
            ax.add_patch(circle)
            
            # Añadir texto del reactivo
            ax.text(i, completitudes[i], f'{reactivos[i]}\n{completitudes[i]:.1f}%', 
                   ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)
            
            # Conectar con línea al siguiente nodo
            if i < len(pasos) - 1:
                ax.arrow(i + 0.3, completitudes[i], 0.4, completitudes[i+1] - completitudes[i], 
                        head_width=2, head_length=0.1, fc='darkgray', ec='darkgray', zorder=2)
        
        # Línea de objetivo (70% completitud)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Objetivo 70%')
        
        ax.set_xlabel('Paso en la Secuencia', fontsize=12, fontweight='bold')
        ax.set_ylabel('Completitud de Habilidades (%)', fontsize=12, fontweight='bold')
        ax.set_title('Evolución de Completitud por Reactivo (Predicción Tipo Árbol)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 110)
        ax.set_xlim(-0.5, len(pasos) - 0.5)
    
    def _graficar_matriz_evolucion_habilidades(self, simulacion: List[Dict], ax):
        """Gráfica matriz de calor mostrando evolución de cada habilidad."""
        ax.clear()
        
        # Preparar datos para matriz de calor
        todas_habilidades = list(simulacion[0]['habilidades'].keys())
        matriz_evolucion = []
        etiquetas_pasos = []
        
        for paso in simulacion:
            fila = [paso['habilidades'][h] for h in todas_habilidades]
            matriz_evolucion.append(fila)
            etiquetas_pasos.append(f"Paso {paso['paso']}: {paso['reactivo']}")
        
        matriz_evolucion = np.array(matriz_evolucion)
        
        # Crear mapa de calor
        im = ax.imshow(matriz_evolucion, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        
        # Configurar ejes
        ax.set_xticks(range(len(todas_habilidades)))
        ax.set_xticklabels(todas_habilidades, rotation=45)
        ax.set_yticks(range(len(etiquetas_pasos)))
        ax.set_yticklabels(etiquetas_pasos)
        
        # Añadir valores en las celdas
        for i in range(len(etiquetas_pasos)):
            for j in range(len(todas_habilidades)):
                value = matriz_evolucion[i, j]
                color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=color, fontsize=10, fontweight='bold')
        
        # Línea de separación en 0.7 (umbral de aprobación)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Calificación de Habilidad', fontsize=12)
        
        ax.set_title('Evolución de Habilidades por Paso (Matriz de Predicción)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Habilidades', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pasos de Aplicación', fontsize=12, fontweight='bold')
    
    def _graficar_impacto_reactivos(self, simulacion: List[Dict], ax):
        """Gráfica el impacto de cada reactivo."""
        ax.clear()
        
        # Filtrar solo los pasos con reactivos (excluir estado inicial)
        pasos_reactivos = [s for s in simulacion if s['paso'] > 0]
        
        if not pasos_reactivos:
            return
        
        reactivos = [s['reactivo'] for s in pasos_reactivos]
        mejoras_promedio = [s['mejora_promedio'] for s in pasos_reactivos]
        habilidades_nuevas_aprobadas = []
        
        # Calcular habilidades nuevas aprobadas en cada paso
        aprobadas_anterior = simulacion[0]['habilidades_aprobadas']
        for s in pasos_reactivos:
            nuevas = s['habilidades_aprobadas'] - aprobadas_anterior
            habilidades_nuevas_aprobadas.append(nuevas)
            aprobadas_anterior = s['habilidades_aprobadas']
        
        # Crear gráfica de barras doble
        x_pos = np.arange(len(reactivos))
        
        bars1 = ax.bar(x_pos - 0.2, mejoras_promedio, 0.4, label='Mejora Promedio', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x_pos + 0.2, habilidades_nuevas_aprobadas, 0.4, 
                      label='Nuevas Habilidades Aprobadas', color='lightgreen', alpha=0.8)
        
        # Añadir valores sobre las barras
        for bar, valor in zip(bars1, mejoras_promedio):
            if valor > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{valor:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar, valor in zip(bars2, habilidades_nuevas_aprobadas):
            if valor > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{valor}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Reactivos Aplicados', fontsize=12, fontweight='bold')
        ax.set_ylabel('Impacto', fontsize=12, fontweight='bold')
        ax.set_title('Impacto Predictivo de Cada Reactivo', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(reactivos, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def graficar_arbol_decision_completo(self):
        """Gráfica un árbol de decisión completo mostrando todas las posibles rutas."""
        if not self.algoritmo.mejor_individuo:
            print("No hay mejor individuo para analizar")
            return
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Crear grafo dirigido
        G = nx.DiGraph()
        
        # Simular múltiples secuencias para crear el árbol
        simulacion_principal = self.algoritmo.simular_mejora_secuencial(self.algoritmo.mejor_individuo)
        
        # Nodo raíz
        nodo_raiz = "Inicio"
        G.add_node(nodo_raiz, level=0, completitud=simulacion_principal[0]['completitud'])
        
        # Agregar nodos y aristas para la secuencia principal
        nodo_anterior = nodo_raiz
        for i, paso in enumerate(simulacion_principal[1:], 1):
            nodo_actual = f"{paso['reactivo']}_paso_{i}"
            G.add_node(nodo_actual, level=i, completitud=paso['completitud'], 
                      reactivo=paso['reactivo'])
            G.add_edge(nodo_anterior, nodo_actual, mejora=paso['mejora_promedio'])
            nodo_anterior = nodo_actual
        
        # Posicionar nodos usando layout jerárquico
        pos = {}
        levels = {}
        
        for node in G.nodes():
            level = G.nodes[node]['level']
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        for level, nodes in levels.items():
            for i, node in enumerate(nodes):
                pos[node] = (level * 2, len(nodes) - i - 1)
        
        # Dibujar el grafo
        ax.clear()
        
        # Dibujar aristas
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True, 
                              arrowsize=20, alpha=0.6, width=2)
        
        # Dibujar nodos con colores según completitud
        for node in G.nodes():
            x, y = pos[node]
            completitud = G.nodes[node]['completitud'] * 100
            
            if completitud >= 70:
                color = 'lightgreen'
            elif completitud >= 50:
                color = 'yellow'
            else:
                color = 'lightcoral'
            
            # Dibujar nodo
            circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, zorder=3)
            ax.add_patch(circle)
            
            # Añadir etiqueta
            if node == "Inicio":
                label = f"Inicio\n{completitud:.1f}%"
            else:
                reactivo = G.nodes[node]['reactivo']
                label = f"{reactivo}\n{completitud:.1f}%"
            
            ax.text(x, y, label, ha='center', va='center', fontsize=9, 
                   fontweight='bold', zorder=4)
        
        # Configurar axes
        ax.set_xlim(-0.5, max(level * 2 for level in levels.keys()) + 0.5)
        ax.set_ylim(-0.5, max(len(nodes) for nodes in levels.values()) + 0.5)
        ax.set_aspect('equal')
        ax.set_title('Árbol de Decisión: Secuencia Óptima de Reactivos', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Leyenda
        legend_elements = [
            mpatches.Patch(color='lightgreen', label='Completitud ≥ 70%'),
            mpatches.Patch(color='yellow', label='Completitud 50-69%'),
            mpatches.Patch(color='lightcoral', label='Completitud < 50%')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def generar_reporte_prediccion_completo(self):
        """Genera el reporte completo con las 3 gráficas de predicción solicitadas."""
        print("="*80)
        print("REPORTE DE PREDICCIÓN SECUENCIAL - EVOLUCIÓN DE HABILIDADES")
        print("="*80)
        
        if not self.algoritmo.mejor_individuo:
            print("No hay mejor individuo para analizar")
            return
        
        # Obtener simulación
        simulacion = self.algoritmo.simular_mejora_secuencial(self.algoritmo.mejor_individuo)
        
        # Mostrar resumen numérico
        print(f"\nSECUENCIA ÓPTIMA DE REACTIVOS:")
        print(f"Reactivos seleccionados: {self.algoritmo.mejor_individuo.genes}")
        
        print(f"\nEVOLUCIÓN PREDICHA:")
        for paso in simulacion:
            if paso['paso'] == 0:
                print(f"Estado inicial: {paso['completitud']*100:.1f}% completitud, "
                      f"{paso['habilidades_aprobadas']} habilidades aprobadas")
            else:
                print(f"Después de {paso['reactivo']}: {paso['completitud']*100:.1f}% completitud, "
                      f"{paso['habilidades_aprobadas']} habilidades aprobadas "
                      f"(+{paso['mejora_promedio']:.3f} mejora promedio)")
        
        print(f"\nGENERANDO GRÁFICAS DE PREDICCIÓN...")
        
        # Generar las 3 gráficas principales
        self.graficar_prediccion_secuencial()
        
        # Generar gráfica adicional de árbol de decisión
        print("Generando árbol de decisión completo...")
        self.graficar_arbol_decision_completo()
        
        print("\n" + "="*80)
        print("REPORTE DE PREDICCIÓN COMPLETADO")
        print("="*80)

    # Métodos originales mantenidos
    def graficar_evolucion_fitness(self):
        """Gráfica la evolución del fitness a lo largo de las generaciones."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.algoritmo.historial_fitness, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title('Evolución del Fitness', fontsize=14, fontweight='bold')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Gráfica de las últimas 10 generaciones con más detalle
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
    
    reactivos_alcanzables = list(reactivos_data.keys())
    
    return reactivos_data, habilidades_data, conteo_reactivos, reactivos_alcanzables

def validar_restricciones():
    """Valida que las restricciones del algoritmo se cumplan."""
    print("Validando restricciones...")
    
    reactivos_data, _, _, _ = crear_datos_ejemplo()
    
    # Verificar que los pesos sumen 1 para cada reactivo
    for reactivo_id, reactivo in reactivos_data.items():
        suma_pesos = sum(reactivo.peso_habilidades.values())
        print(f"Reactivo {reactivo_id}: Suma de pesos = {suma_pesos:.6f}")
        assert abs(suma_pesos - 1.0) < 1e-6, f"Los pesos del reactivo {reactivo_id} no suman 1"
    
    print("✓ Todas las restricciones se cumplen correctamente")

class ConfiguradorParametros:
    """Clase para ayudar a configurar los parámetros del algoritmo genético."""
    
    @staticmethod
    def recomendar_parametros(num_reactivos: int, num_habilidades: int, 
                            complejidad_problema: str = "medio") -> Dict:
        """Recomienda parámetros basados en las características del problema."""
        
        configuraciones = {
            "simple": {
                "K": min(3, num_reactivos // 3),
                "tamaño_poblacion": 15,
                "generaciones": 30,
                "tasa_mutacion": 0.1,
                "presion_seleccion": 0.6
            },
            "medio": {
                "K": min(5, num_reactivos // 2),
                "tamaño_poblacion": 20,
                "generaciones": 50,
                "tasa_mutacion": 0.15,
                "presion_seleccion": 0.7
            },
            "complejo": {
                "K": min(7, num_reactivos),
                "tamaño_poblacion": 30,
                "generaciones": 100,
                "tasa_mutacion": 0.2,
                "presion_seleccion": 0.8
            }
        }
        
        config = configuraciones.get(complejidad_problema, configuraciones["medio"])
        
        # Ajustes automáticos basados en el tamaño del problema
        if num_reactivos < 10:
            config["tamaño_poblacion"] = max(10, config["tamaño_poblacion"] // 2)
        elif num_reactivos > 50:
            config["tamaño_poblacion"] = min(50, config["tamaño_poblacion"] * 2)
        
        return config

def ejecutar_ejemplo_completo():
    """Ejecuta un ejemplo completo del algoritmo genético con todas las visualizaciones."""
    print("Creando datos de ejemplo...")
    reactivos_data, habilidades_data, conteo_reactivos, reactivos_alcanzables = crear_datos_ejemplo()
    
    print("Inicializando algoritmo genético...")
    ag = AlgoritmoGenetico(
        reactivos_alcanzables=reactivos_alcanzables,
        reactivos_data=reactivos_data,
        habilidades_data=habilidades_data,
        conteo_reactivos=conteo_reactivos,
        K=3,
        tamaño_poblacion=20,
        generaciones=50,
        tasa_mutacion=0.15,
        presion_seleccion=0.7
    )
    
    print("Ejecutando evolución...")
    ag.evolucionar()
    
    print("Generando visualizaciones y reportes...")
    visualizador = VisualizadorResultados(ag)
    
    # Generar TODAS las visualizaciones (originales + nuevas)
    print("\n--- REPORTE ORIGINAL ---")
    visualizador.generar_reporte_completo()
    
    print("\n--- REPORTE DE PREDICCIÓN ---")
    visualizador.generar_reporte_prediccion_completo()
    
    return ag, visualizador

def ejecutar_ejemplo_con_prediccion():
    """Ejecuta un ejemplo completo del algoritmo genético con predicciones."""
    return ejecutar_ejemplo_completo()

# Función principal para ejecutar el algoritmo
if __name__ == "__main__":
    print("="*80)
    print("ALGORITMO GENÉTICO PARA SELECCIÓN DE REACTIVOS EDUCATIVOS")
    print("CON GRÁFICAS DE PREDICCIÓN SECUENCIAL")
    print("="*80)
    
    validar_restricciones()
    
    # Ejecutar ejemplo completo con TODAS las visualizaciones
    algoritmo, visualizador = ejecutar_ejemplo_completo()
    
    print("\n" + "="*80)
    print("EJECUCIÓN COMPLETADA - TODAS LAS GRÁFICAS GENERADAS")
    print("="*80)
    
    print("\nGRÁFICAS GENERADAS:")
    print("✓ Gráficas Originales:")
    print("  - Evolución del Fitness")
    print("  - Métricas de Objetivos")
    print("  - Estado de Habilidades")
    print("  - Análisis de Reactivos Seleccionados")
    print("✓ Nuevas Gráficas de Predicción:")
    print("  - Evolución de Completitud (Tipo Árbol)")
    print("  - Matriz de Evolución de Habilidades")
    print("  - Impacto de Reactivos")
    print("  - Árbol de Decisión Completo")
    
    print("\nRECOMENDACIONES DE PARÁMETROS:")
    config_simple = ConfiguradorParametros.recomendar_parametros(10, 8, "simple")
    config_medio = ConfiguradorParametros.recomendar_parametros(10, 8, "medio")
    config_complejo = ConfiguradorParametros.recomendar_parametros(10, 8, "complejo")
    
    print("Problema simple:", config_simple)
    print("Problema medio:", config_medio)
    print("Problema complejo:", config_complejo)
    
    print("\nUSO ADICIONAL:")
    print("# Para generar solo gráficas originales:")
    print("visualizador.generar_reporte_completo()")
    print("\n# Para generar solo gráficas de predicción:")
    print("visualizador.generar_reporte_prediccion_completo()")