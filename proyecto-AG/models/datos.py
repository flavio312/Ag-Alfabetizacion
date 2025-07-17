from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Reactivo:
    """Representa un reactivo educativo con sus habilidades asociadas."""
    id: str
    habilidades: List[str]
    peso_habilidades: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.peso_habilidades:
            # Distribución equitativa si no hay pesos especificados
            peso_igual = 1.0 / len(self.habilidades)
            self.peso_habilidades = {h: peso_igual for h in self.habilidades}
        else:
            # Normalización de pesos existentes
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

    def mejorar(self, incremento: float):
        """Mejora la calificación de la habilidad."""
        self.calificacion = min(1.0, self.calificacion + incremento)
        self.aprobada = self.calificacion >= self.umbral_aprobacion