from typing import Dict

class ConfiguradorParametros:
    """Ayuda a configurar los parámetros del algoritmo genético."""
    
    @staticmethod
    def recomendar_parametros(num_reactivos: int, num_habilidades: int, 
                            complejidad_problema: str = "medio") -> Dict:
        
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
        
        # Ajustes basados en el tamaño del problema
        if num_reactivos < 10:
            config["tamaño_poblacion"] = max(10, config["tamaño_poblacion"] // 2)
        elif num_reactivos > 50:
            config["tamaño_poblacion"] = min(50, config["tamaño_poblacion"] * 2)
        
        return config