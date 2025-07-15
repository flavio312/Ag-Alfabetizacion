from app import ejecutar_algoritmo_completo, comparar_estrategias
from src.models.individual.individual import Individual
from src.models.environment.environment import Environment
from src.test.validation import validar_estrategia

def ejemplo_basico():
    print("=== EJEMPLO BÁSICO ===")
    
    # Crear un individuo manualmente
    individuo_manual = Individual(["R1", "R9", "R15"])
    print(f"Individuo manual: {individuo_manual}")
    print(f"Fitness: {individuo_manual.fitness:.4f}")
    
    # Validar la estrategia
    resultado = validar_estrategia(individuo_manual, mostrar_detalle=True)
    print(f"Mejora potencial en fitness: {resultado['mejora_fitness']:+.4f}")

def ejemplo_personalizado():
    """Ejemplo con parámetros personalizados"""
    print("\n=== EJEMPLO CON PARÁMETROS PERSONALIZADOS ===")
    
    # Crear población inicial pequeña
    poblacion = [
        Individual(["R1", "R2", "R3"]),
        Individual(["R4", "R5", "R6"]),
        Individual(["R7", "R8", "R9"]),
        Individual(["R10", "R11", "R12"])
    ]
    
    # Configurar ambiente
    ambiente = Environment(
        poblacion=poblacion,
        generations=2,
        K=3,
        mutation_rate=0.2,  # Mayor mutación
        selection_pressure=0.8  # Mayor presión de selección
    )
    
    # Ejecutar
    ambiente.start()
    
    # Obtener mejor resultado
    mejor = ambiente.get_best_individual()
    print(f"\nMejor individuo: {mejor.gens}")
    print(f"Fitness: {mejor.fitness:.4f}")

def ejemplo_completo_con_analisis():
    """Ejemplo completo con análisis detallado"""
    print("\n=== EJEMPLO COMPLETO CON ANÁLISIS ===")
    
    # Ejecutar algoritmo completo
    ambiente, resultados_validacion = ejecutar_algoritmo_completo()
    
    # Análisis adicional
    print(f"\n--- ANÁLISIS ADICIONAL ---")
    
    # Top 3 individuos por métrica específica
    print(f"\nTop 3 por OBJ1 (habilidades no aprobadas):")
    por_obj1 = sorted(ambiente.poblacion, key=lambda x: x.metrica_1, reverse=True)
    for i, ind in enumerate(por_obj1[:3]):
        print(f"  {i+1}. {ind.gens} - OBJ1: {ind.metrica_1:.3f}")
    
    print(f"\nTop 3 por OBJ4 (cobertura de habilidades):")
    por_obj4 = sorted(ambiente.poblacion, key=lambda x: x.metrica_4, reverse=True)
    for i, ind in enumerate(por_obj4[:3]):
        print(f"  {i+1}. {ind.gens} - OBJ4: {ind.metrica_4}")
    
    # Comparar con validación
    mejor_actual = max(ambiente.poblacion, key=lambda x: x.fitness)
    mejor_validado = max(resultados_validacion, key=lambda x: x["despues"]["fitness"])
    
    print(f"\nComparación final:")
    print(f"  Mejor actual: {mejor_actual.gens} (Fitness: {mejor_actual.fitness:.4f})")
    print(f"  Mejor validado: {mejor_validado['individuo'].gens}")
    print(f"  Fitness potencial: {mejor_validado['despues']['fitness']:.4f}")
    print(f"  Ganancia esperada: {mejor_validado['mejora_fitness']:+.4f}")

def experimento_parametros():
    """Experimenta con diferentes parámetros"""
    print("\n=== EXPERIMENTO CON PARÁMETROS ===")
    
    configuraciones = [
        {"mut": 0.05, "sel": 0.3, "gen": 3, "desc": "Baja mutación, baja selección"},
        {"mut": 0.15, "sel": 0.6, "gen": 3, "desc": "Mutación media, selección media"},
        {"mut": 0.25, "sel": 0.9, "gen": 3, "desc": "Alta mutación, alta selección"}
    ]
    
    for config in configuraciones:
        print(f"\n--- {config['desc']} ---")
        
        # Crear población base (misma para todas las pruebas)
        from app import crear_poblacion_inicial
        poblacion = crear_poblacion_inicial(8, 3)
        
        ambiente = Environment(
            poblacion=poblacion,
            generations=config["gen"],
            mutation_rate=config["mut"],
            selection_pressure=config["sel"]
        )
        
        ambiente.start()
        mejor = ambiente.get_best_individual()
        
        print(f"Mejor fitness: {mejor.fitness:.4f}")
        print(f"Mejor genes: {mejor.gens}")

if __name__ == "__main__":
    # Ejecutar ejemplos
    ejemplo_basico()
    ejemplo_personalizado()
    ejemplo_completo_con_analisis()