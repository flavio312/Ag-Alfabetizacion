# src/models/individual/individual.py
from typing import List, Set
from ...test.main_model_SQL import hab, reactivos, mostrar_tabla_de, reactivos_realizados

class Individual:
    def __init__(self, gens: List[str]):
        self.gens = gens.copy()  # Usar copia para evitar modificaciones accidentales
        self.show_table = False
        self.update_individual()
    
    def update_individual(self):
        """Actualiza todas las métricas y fitness del individuo"""
        self._calculate_habilidades()
        self._calculate_metricas()
        self._calculate_fitness()
        self._update_data()
    
    def _calculate_habilidades(self):
        """Calcula las habilidades aprobadas y no aprobadas"""
        self.habs_no_aprob = []
        self.habs_aprob = []
        
        # Obtener todas las habilidades únicas involucradas en los reactivos
        habilidades_involucradas = set()
        for reactivo in self.gens:
            habilidades_involucradas.update(reactivos[reactivo])
        
        # Clasificar habilidades por estado de aprobación
        for habilidad in habilidades_involucradas:
            calificacion = hab[habilidad]
            if calificacion < 0.7:
                self.habs_no_aprob.append(habilidad)
            else:
                self.habs_aprob.append(habilidad)
    
    def _calculate_metricas(self):
        """Calcula las 4 métricas según la especificación"""
        # OBJ1: Maximizar uso de reactivos con habilidades no aprobatorias
        total_habilidades = len(self.habs_aprob) + len(self.habs_no_aprob)
        if total_habilidades > 0:
            self.metrica_1 = len(self.habs_no_aprob) / total_habilidades
        else:
            self.metrica_1 = 0.0
        
        # OBJ2: Minimizar uso de reactivos ya realizados
        self.metrica_2 = sum(reactivos_realizados.get(reactivo, 0) for reactivo in self.gens)
        
        # OBJ3: Minimizar uso de reactivos con habilidades ya aprobadas
        self.metrica_3 = self._count_reactivos_con_habilidades_aprobadas()
        
        # OBJ4: Maximizar cantidad de habilidades involucradas
        habilidades_totales = set()
        for reactivo in self.gens:
            habilidades_totales.update(reactivos[reactivo])
        self.metrica_4 = len(habilidades_totales)
    
    def _count_reactivos_con_habilidades_aprobadas(self) -> int:
        """Cuenta reactivos que tienen al menos una habilidad aprobada"""
        count = 0
        for reactivo in self.gens:
            habilidades_reactivo = reactivos[reactivo]
            if any(hab[h] >= 0.7 for h in habilidades_reactivo):
                count += 1
        return count
    
    def _calculate_fitness(self):
        """Calcula el fitness según la fórmula especificada"""
        # Fitness = [(1 + OBJ1) * (1 + OBJ4)] / [(1 + OBJ2) * (1 + OBJ3)]
        numerador = (1 + self.metrica_1) * (1 + self.metrica_4)
        denominador = (1 + self.metrica_2) * (1 + self.metrica_3)
        
        if denominador == 0:
            self.fitness = 0.0
        else:
            self.fitness = numerador / denominador
    
    def _update_data(self):
        """Actualiza el diccionario de datos para reportes"""
        self.data = {
            "HNA": len(self.habs_no_aprob),
            "HA": len(self.habs_aprob),
            "Metricas": {
                "Metrica_1_OBJ1": self.metrica_1,
                "Metrica_2_OBJ2": self.metrica_2,
                "Metrica_3_OBJ3": self.metrica_3,
                "Metrica_4_OBJ4": self.metrica_4
            },
            "Fitness": self.fitness
        }
    
    def get_habilidades_involucradas(self) -> Set[str]:
        """Retorna el conjunto de habilidades involucradas en los reactivos"""
        habilidades = set()
        for reactivo in self.gens:
            habilidades.update(reactivos[reactivo])
        return habilidades
    
    def get_reactivos_completamente_aprobados(self) -> List[str]:
        """Retorna lista de reactivos donde todas sus habilidades están aprobadas"""
        reactivos_aprobados = []
        for reactivo in self.gens:
            habilidades_reactivo = reactivos[reactivo]
            if all(hab[h] >= 0.7 for h in habilidades_reactivo):
                reactivos_aprobados.append(reactivo)
        return reactivos_aprobados
    
    def copy(self):
        """Crea una copia del individuo"""
        return Individual(self.gens.copy())
    
    def __str__(self):
        if self.show_table:
            tabla = mostrar_tabla_de(self.gens)
            return f"{tabla}\n" \
                   f"Genes: {self.gens}\n" \
                   f"Fitness: {self.fitness:.4f}\n" \
                   f"Métricas - OBJ1: {self.metrica_1:.3f}, OBJ2: {self.metrica_2}, " \
                   f"OBJ3: {self.metrica_3}, OBJ4: {self.metrica_4}\n" \
                   f"Habilidades no aprobadas: {len(self.habs_no_aprob)}, " \
                   f"Aprobadas: {len(self.habs_aprob)}\n"
        else:
            return f"Genes: {self.gens} | Fitness: {self.fitness:.4f} | " \
                   f"HNA: {len(self.habs_no_aprob)} | HA: {len(self.habs_aprob)}"
    
    def __repr__(self):
        return f"Individual(gens={self.gens}, fitness={self.fitness:.4f})"