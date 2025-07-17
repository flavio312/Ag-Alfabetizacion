from typing import List, Tuple
import random
from models.individuo import Individuo

class OperadoresGeneticos:
    """Contiene los operadores genéticos para el algoritmo."""
    
    def __init__(self, reactivos_alcanzables: List[str], tasa_mutacion: float, presion_seleccion: float):
        self.reactivos_alcanzables = reactivos_alcanzables
        self.tasa_mutacion = tasa_mutacion
        self.presion_seleccion = presion_seleccion
    
    def seleccionar_padres(self, poblacion: List[Individuo]) -> Tuple[Individuo, Individuo]:
        """Selección por torneo."""
        tamaño_torneo = max(2, int(len(poblacion) * self.presion_seleccion))
        
        # Seleccionar padre 1
        candidatos1 = random.sample(poblacion, min(tamaño_torneo, len(poblacion)))
        padre1 = max(candidatos1, key=lambda x: x.fitness)
        
        # Seleccionar padre 2 (diferente al padre 1)
        candidatos2 = [ind for ind in poblacion if ind != padre1]
        if candidatos2:
            candidatos2 = random.sample(candidatos2, min(tamaño_torneo, len(candidatos2)))
            padre2 = max(candidatos2, key=lambda x: x.fitness)
        else:
            padre2 = padre1
        
        return padre1, padre2
    
    def cruzar(self, padre1: Individuo, padre2: Individuo) -> Tuple[List[str], List[str]]:
        """Cruza con corrección de duplicados."""
        punto_cruza = random.randint(1, len(padre1.genes) - 1)
        
        # Crear hijos iniciales
        hijo1_genes = padre1.genes[:punto_cruza] + padre2.genes[punto_cruza:]
        hijo2_genes = padre2.genes[:punto_cruza] + padre1.genes[punto_cruza:]
        
        # Corregir duplicados
        hijo1_genes = self._corregir_duplicados(hijo1_genes, padre1.genes + padre2.genes)
        hijo2_genes = self._corregir_duplicados(hijo2_genes, padre1.genes + padre2.genes)
        
        return hijo1_genes, hijo2_genes
    
    def _corregir_duplicados(self, genes: List[str], genes_padres: List[str]) -> List[str]:
        """Corrige duplicados en los genes."""
        genes_corregidos = genes.copy()
        todos_reactivos = set(self.reactivos_alcanzables)
        reactivos_en_genes = set(genes_corregidos)
        reactivos_faltantes = list(todos_reactivos - reactivos_en_genes)
        
        i = 0
        while i < len(genes_corregidos) and reactivos_faltantes:
            reactivo_actual = genes_corregidos[i]
            ocurrencias = [j for j in range(i+1, len(genes_corregidos)) 
                          if genes_corregidos[j] == reactivo_actual]
            
            if ocurrencias:
                pos_duplicado = random.choice(ocurrencias)
                reactivo_reemplazo = random.choice(reactivos_faltantes)
                genes_corregidos[pos_duplicado] = reactivo_reemplazo
                reactivos_faltantes.remove(reactivo_reemplazo)
            
            i += 1
        
        return genes_corregidos
    
    def mutar(self, genes: List[str]) -> List[str]:
        """Aplica mutación a un conjunto de genes."""
        if random.random() < self.tasa_mutacion:
            genes_mutados = genes.copy()
            pos = random.randint(0, len(genes_mutados) - 1)
            
            reactivos_disponibles = [r for r in self.reactivos_alcanzables 
                                   if r not in genes_mutados]
            
            if reactivos_disponibles:
                nuevo_reactivo = random.choice(reactivos_disponibles)
                genes_mutados[pos] = nuevo_reactivo
            
            return genes_mutados
        
        return genes.copy()
    
    def seleccionar_supervivientes(self, poblacion_actual: List[Individuo], 
                                 descendencia: List[Individuo], tamaño_poblacion: int) -> List[Individuo]:
        """Selección elitista con diversidad."""
        poblacion_expandida = poblacion_actual + descendencia
        poblacion_ordenada = sorted(poblacion_expandida, key=lambda x: x.fitness, reverse=True)
        
        tamaño_elite = tamaño_poblacion // 2
        elite = poblacion_ordenada[:tamaño_elite]
        
        tamaño_aleatorio = tamaño_poblacion // 4
        resto = poblacion_ordenada[tamaño_elite:]
        
        aleatorios = random.sample(resto, min(tamaño_aleatorio, len(resto))) if resto and tamaño_aleatorio > 0 else []
        
        nueva_poblacion = elite + aleatorios
        while len(nueva_poblacion) < tamaño_poblacion:
            nueva_poblacion.append(random.choice(poblacion_ordenada[:tamaño_poblacion]))
        
        return nueva_poblacion