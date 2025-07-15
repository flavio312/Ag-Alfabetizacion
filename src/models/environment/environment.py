# src/models/environment/environment.py
from random import sample, randint, random, choice
from typing import List, Tuple
from ..individual.individual import Individual
from ...test.main_model_SQL import reactivos

class Environment:
    def __init__(self, poblacion: List[Individual], generations: int = 1, 
                 K: int = 3, mutation_rate: float = 0.1, selection_pressure: float = 0.5):
        """
        Inicializa el ambiente del algoritmo genético
        
        Args:
            poblacion: Lista de individuos iniciales
            generations: Número de generaciones a ejecutar
            K: Parámetro del sistema para los primeros K reactivos
            mutation_rate: Probabilidad de mutación (0.0 a 1.0)
            selection_pressure: Presión de selección (0.0 a 1.0)
        """
        self.poblacion = poblacion
        self.generations = generations
        self.K = K
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.all_reactivos = list(reactivos.keys())
        self.best_individuals_history = []  # Historial de mejores individuos
        
    def start(self):
        """Ejecuta el algoritmo genético"""
        print(f"Iniciando algoritmo genético con {len(self.poblacion)} individuos")
        print(f"Parámetros: K={self.K}, generaciones={self.generations}, "
              f"mutación={self.mutation_rate}, selección={self.selection_pressure}")
        
        for generation in range(self.generations):
            print(f"\n--- Generación {generation + 1} ---")
            
            # Guardar mejor individuo de la generación
            mejor_actual = max(self.poblacion, key=lambda x: x.fitness)
            self.best_individuals_history.append(mejor_actual.copy())
            
            # Mostrar estadísticas de la generación
            self._show_generation_stats(generation + 1)
            
            # Realizar cruza y generar nueva población
            self.crosses()
            
            # Aplicar mutación
            self.mutate()
            
            # Selección de supervivientes
            self.selection()
        
        print(f"\nAlgoritmo genético completado después de {self.generations} generaciones")
        self._show_final_stats()
    
    def select_pair(self) -> Tuple[Individual, Individual]:
        """Selecciona un par de padres usando selección por torneo"""
        tournament_size = max(2, int(len(self.poblacion) * self.selection_pressure))
        
        # Torneo para padre 1
        candidates1 = sample(self.poblacion, min(tournament_size, len(self.poblacion)))
        padre = max(candidates1, key=lambda x: x.fitness)
        
        # Torneo para padre 2 (asegurar que sea diferente)
        candidates2 = sample(self.poblacion, min(tournament_size, len(self.poblacion)))
        madre = max(candidates2, key=lambda x: x.fitness)
        
        # Si son el mismo individuo, seleccionar otro
        if padre is madre and len(self.poblacion) > 1:
            remaining = [ind for ind in self.poblacion if ind is not padre]
            madre = choice(remaining)
        
        return padre, madre
    
    def crosses(self):
        """Realiza cruza con corrección de duplicados según la estrategia especificada"""
        nuevos_individuos = []
        
        # Generar tantos hijos como individuos actuales
        for _ in range(len(self.poblacion)):
            padre, madre = self.select_pair()
            
            # Punto de cruza aleatorio
            punto_cruza = randint(1, len(padre.gens) - 1)
            
            # Crear hijos iniciales
            hijo1_genes = padre.gens[:punto_cruza] + madre.gens[punto_cruza:]
            hijo2_genes = madre.gens[:punto_cruza] + padre.gens[punto_cruza:]
            
            # Corregir duplicados usando la estrategia especificada
            hijo1_genes = self._corregir_duplicados(hijo1_genes, padre.gens + madre.gens)
            hijo2_genes = self._corregir_duplicados(hijo2_genes, padre.gens + madre.gens)
            
            # Crear individuos
            nuevos_individuos.append(Individual(hijo1_genes))
            nuevos_individuos.append(Individual(hijo2_genes))
        
        # Agregar nuevos individuos a la población
        self.poblacion.extend(nuevos_individuos)
    
    def _corregir_duplicados(self, genes: List[str], genes_padres: List[str]) -> List[str]:
        """
        Corrige duplicados y faltantes en los genes según la estrategia especificada
        """
        genes_corregidos = genes.copy()
        todos_reactivos = set(self.all_reactivos)
        reactivos_padres = set(genes_padres)
        
        # Paso 1: Identificar reactivos faltantes
        reactivos_en_hijo = set(genes_corregidos)
        faltantes = reactivos_padres - reactivos_en_hijo
        
        if not faltantes:
            return genes_corregidos
        
        # Paso 2: Corregir duplicados
        posiciones_visitadas = set()
        
        for i in range(len(genes_corregidos)):
            if i in posiciones_visitadas:
                continue
                
            reactivo_actual = genes_corregidos[i]
            
            # Buscar duplicados de este reactivo
            duplicados = []
            for j in range(i + 1, len(genes_corregidos)):
                if genes_corregidos[j] == reactivo_actual:
                    duplicados.append(j)
            
            # Si hay duplicados, reemplazar uno aleatoriamente
            if duplicados and faltantes:
                pos_a_cambiar = choice(duplicados)
                reactivo_faltante = choice(list(faltantes))
                
                genes_corregidos[pos_a_cambiar] = reactivo_faltante
                faltantes.remove(reactivo_faltante)
                posiciones_visitadas.add(pos_a_cambiar)
                
                if not faltantes:
                    break
        
        return genes_corregidos
    
    def mutate(self):
        """Aplica mutación a algunos individuos"""
        for individuo in self.poblacion:
            if random() < self.mutation_rate:
                # Seleccionar posición aleatoria para mutar
                pos = randint(0, len(individuo.gens) - 1)
                
                # Seleccionar nuevo reactivo que no esté ya en el individuo
                reactivos_disponibles = [r for r in self.all_reactivos if r not in individuo.gens]
                
                if reactivos_disponibles:
                    nuevo_reactivo = choice(reactivos_disponibles)
                    individuo.gens[pos] = nuevo_reactivo
                    individuo.update_individual()
    
    def selection(self):
        """Selecciona supervivientes para la siguiente generación"""
        # Ordenar por fitness (descendente)
        self.poblacion.sort(key=lambda x: x.fitness, reverse=True)
        
        # Mantener solo la mitad superior + algunos aleatorios
        tamaño_elite = len(self.poblacion) // 2
        tamaño_aleatorio = len(self.poblacion) // 4
        
        # Elite (mejores individuos)
        elite = self.poblacion[:tamaño_elite]
        
        # Algunos aleatorios del resto para mantener diversidad
        resto = self.poblacion[tamaño_elite:]
        if resto and tamaño_aleatorio > 0:
            aleatorios = sample(resto, min(tamaño_aleatorio, len(resto)))
        else:
            aleatorios = []
        
        # Nueva población
        self.poblacion = elite + aleatorios
    
    def _show_generation_stats(self, generation: int):
        """Muestra estadísticas de la generación actual"""
        fitness_values = [ind.fitness for ind in self.poblacion]
        
        mejor = max(self.poblacion, key=lambda x: x.fitness)
        peor = min(self.poblacion, key=lambda x: x.fitness)
        promedio = sum(fitness_values) / len(fitness_values)
        
        print(f"Población: {len(self.poblacion)} individuos")
        print(f"Fitness - Mejor: {mejor.fitness:.4f}, Peor: {peor.fitness:.4f}, "
              f"Promedio: {promedio:.4f}")
        print(f"Mejor individuo: {mejor.gens}")
    
    def _show_final_stats(self):
        """Muestra estadísticas finales del algoritmo"""
        print("\n" + "="*60)
        print("ESTADÍSTICAS FINALES")
        print("="*60)
        
        mejor_final = max(self.poblacion, key=lambda x: x.fitness)
        mejor_historico = max(self.best_individuals_history, key=lambda x: x.fitness)
        
        print(f"Mejor individuo final: {mejor_final.gens}")
        print(f"Fitness final: {mejor_final.fitness:.4f}")
        print(f"Habilidades no aprobadas: {len(mejor_final.habs_no_aprob)}")
        print(f"Habilidades aprobadas: {len(mejor_final.habs_aprob)}")
        
        if mejor_historico.fitness > mejor_final.fitness:
            print(f"\nMejor individuo histórico: {mejor_historico.gens}")
            print(f"Fitness histórico: {mejor_historico.fitness:.4f}")
        
        # Mostrar evolución del fitness
        print(f"\nEvolución del mejor fitness por generación:")
        for i, ind in enumerate(self.best_individuals_history):
            print(f"  Gen {i+1}: {ind.fitness:.4f}")
    
    def get_best_individual(self) -> Individual:
        """Retorna el mejor individuo de la población actual"""
        return max(self.poblacion, key=lambda x: x.fitness)
    
    def get_best_historical(self) -> Individual:
        """Retorna el mejor individuo de toda la historia"""
        if self.best_individuals_history:
            return max(self.best_individuals_history, key=lambda x: x.fitness)
        return self.get_best_individual()
    
    def print_pob(self, show_table: bool = False):
        """Imprime la población actual"""
        print(f"\n--- POBLACIÓN ACTUAL ({len(self.poblacion)} individuos) ---")
        for i, individuo in enumerate(self.poblacion):
            individuo.show_table = show_table
            print(f"{i+1:2d}. {individuo}")
        print("-" * 50)