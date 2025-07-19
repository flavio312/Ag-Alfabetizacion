from typing import List, Dict, Optional
from models.individuo import Individuo
from models.datos import Reactivo, Habilidad
from algoritmo.operadores import OperadoresGeneticos
import random

class AlgoritmoGenetico:    
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
        
        self.operadores = OperadoresGeneticos(
            reactivos_alcanzables=reactivos_alcanzables,
            tasa_mutacion=tasa_mutacion,
            presion_seleccion=presion_seleccion
        )
        
        self.poblacion: List[Individuo] = []
        self.mejor_individuo: Optional[Individuo] = None
        self.historial_fitness: List[float] = []
        self.historial_metricas: List[Dict] = []
    
    def inicializar_poblacion(self):
        """Crea la población inicial."""
        self.poblacion = []
        
        for _ in range(self.tamaño_poblacion):
            genes = random.sample(self.reactivos_alcanzables, 
                                min(self.K, len(self.reactivos_alcanzables)))
            while len(genes) < self.K:
                reactivo_extra = random.choice(self.reactivos_alcanzables)
                if reactivo_extra not in genes:
                    genes.append(reactivo_extra)
            
            individuo = Individuo(genes, self.K, self.reactivos_data, 
                                self.habilidades_data, self.conteo_reactivos)
            self.poblacion.append(individuo)
    
    def evolucionar(self):
        """Ejecuta el proceso evolutivo."""
        self.inicializar_poblacion()
        
        for generacion in range(self.generaciones):
            self._registrar_estadisticas(generacion)
            descendencia = self._generar_descendencia()
            self._seleccionar_supervivientes(descendencia)
    
    def _registrar_estadisticas(self, generacion: int):
        """Registra las estadísticas de la generación actual."""
        mejor_actual = max(self.poblacion, key=lambda x: x.fitness)
        self.historial_fitness.append(mejor_actual.fitness)
        self.historial_metricas.append(mejor_actual.metricas.copy())
        
        if self.mejor_individuo is None or mejor_actual.fitness > self.mejor_individuo.fitness:
            self.mejor_individuo = mejor_actual.copy()
        
        if generacion % 10 == 0:
            print(f"Generación {generacion}: Mejor fitness = {mejor_actual.fitness:.4f}")
    
    def _generar_descendencia(self) -> List[Individuo]:
        """Genera la descendencia para la siguiente generación."""
        descendencia = []
        
        while len(descendencia) < self.tamaño_poblacion:
            padre1, padre2 = self.operadores.seleccionar_padres(self.poblacion)
            hijo1_genes, hijo2_genes = self.operadores.cruzar(padre1, padre2)
            
            hijo1 = Individuo(
                self.operadores.mutar(hijo1_genes),
                self.K, self.reactivos_data, 
                self.habilidades_data, self.conteo_reactivos
            )
            
            hijo2 = Individuo(
                self.operadores.mutar(hijo2_genes),
                self.K, self.reactivos_data, 
                self.habilidades_data, self.conteo_reactivos
            )
            
            descendencia.extend([hijo1, hijo2])
        
        return descendencia
    
    def _seleccionar_supervivientes(self, descendencia: List[Individuo]):
        """Selecciona los individuos que pasan a la siguiente generación."""
        self.poblacion = self.operadores.seleccionar_supervivientes(
            self.poblacion, descendencia, self.tamaño_poblacion
        )