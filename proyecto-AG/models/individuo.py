from typing import List, Dict, Optional
from copy import deepcopy
from models.datos import Reactivo, Habilidad

class Individuo:
    """Representa una solución potencial en el algoritmo genético."""
    
    def __init__(self, genes: List[str], K: int, reactivos_data: Dict[str, Reactivo], 
                 habilidades_data: Dict[str, Habilidad], conteo_reactivos: Dict[str, int]):
        self.genes = genes[:K]
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
            if any(self.habilidades_data[h].aprobada for h in reactivo.habilidades):
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
        return Individuo(
            genes=self.genes.copy(),
            K=self.K,
            reactivos_data=self.reactivos_data,
            habilidades_data=self.habilidades_data,
            conteo_reactivos=self.conteo_reactivos
        )
    
    def simular_mejora(self) -> Dict:
        """Simula la mejora esperada al resolver los reactivos."""
        habilidades_temp = deepcopy(self.habilidades_data)
        
        for reactivo_id in self.genes:
            reactivo = self.reactivos_data[reactivo_id]
            for habilidad_id, peso in reactivo.peso_habilidades.items():
                habilidad = habilidades_temp[habilidad_id]
                if habilidad.calificacion < 0.7:
                    habilidad.mejorar(peso * 0.3)
        
        individuo_temp = Individuo(
            genes=self.genes.copy(),
            K=self.K,
            reactivos_data=self.reactivos_data,
            habilidades_data=habilidades_temp,
            conteo_reactivos=self.conteo_reactivos
        )
        
        habilidades_antes = sum(1 for h in self.habilidades_data.values() if h.aprobada)
        habilidades_despues = sum(1 for h in habilidades_temp.values() if h.aprobada)
        
        return {
            'fitness_antes': self.fitness,
            'fitness_despues': individuo_temp.fitness,
            'mejora_fitness': individuo_temp.fitness - self.fitness,
            'habilidades_aprobadas_antes': habilidades_antes,
            'habilidades_aprobadas_despues': habilidades_despues,
            'nuevas_habilidades_aprobadas': habilidades_despues - habilidades_antes
        }