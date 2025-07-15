from random import sample, seed
from typing import List
from src.models.individual.individual import Individual
from src.models.environment.environment import Environment
from src.test.main_model_SQL import mostrar_tabla_MRH, MRH, reactivos, hab
from src.test.validation import validar_estrategia, validar_poblacion, mostrar_reporte_validacion

def crear_poblacion_inicial(tamaño_poblacion: int = 10, K: int = 3) -> List[Individual]:
    """
    Crea una población inicial aleatoria
    
    Args:
        tamaño_poblacion: Número de individuos en la población
        K: Número de reactivos por individuo
        
    Returns:
        Lista de individuos iniciales
    """
    poblacion = []
    reactivos_disponibles = list(reactivos.keys())
    
    for i in range(tamaño_poblacion):
        # Seleccionar K reactivos aleatorios sin repetición
        genes = sample(reactivos_disponibles, K)
        individuo = Individual(genes)
        poblacion.append(individuo)
    
    return poblacion

def crear_poblacion_diversa(tamaño_poblacion: int = 10, K: int = 3) -> List[Individual]:
    """
    Crea una población inicial más diversa usando diferentes estrategias
    """
    poblacion = []
    reactivos_disponibles = list(reactivos.keys())
    
    # 30% - Individuos completamente aleatorios
    for _ in range(int(tamaño_poblacion * 0.3)):
        genes = sample(reactivos_disponibles, K)
        poblacion.append(Individual(genes))
    
    # 30% - Individuos enfocados en habilidades no aprobadas
    habilidades_bajas = [h for h, cal in hab.items() if cal < 0.7]
    reactivos_habilidades_bajas = []
    for r, habs in reactivos.items():
        if any(h in habilidades_bajas for h in habs):
            reactivos_habilidades_bajas.append(r)
    
    for _ in range(int(tamaño_poblacion * 0.3)):
        if len(reactivos_habilidades_bajas) >= K:
            genes = sample(reactivos_habilidades_bajas, K)
        else:
            genes = sample(reactivos_disponibles, K)
        poblacion.append(Individual(genes))
    
    # 20% - Individuos que maximizan cobertura de habilidades
    for _ in range(int(tamaño_poblacion * 0.2)):
        genes = seleccionar_maxima_cobertura(K)
        poblacion.append(Individual(genes))
    
    # 20% - Completar con aleatorios
    while len(poblacion) < tamaño_poblacion:
        genes = sample(reactivos_disponibles, K)
        poblacion.append(Individual(genes))
    
    return poblacion

def seleccionar_maxima_cobertura(K: int) -> List[str]:
    """
    Selecciona K reactivos que maximizan la cobertura de habilidades
    """
    reactivos_disponibles = list(reactivos.keys())
    genes_seleccionados = []
    habilidades_cubiertas = set()
    
    for _ in range(K):
        mejor_reactivo = None
        mejor_nuevas_habilidades = 0
        
        for reactivo in reactivos_disponibles:
            if reactivo not in genes_seleccionados:
                nuevas_habilidades = set(reactivos[reactivo]) - habilidades_cubiertas
                if len(nuevas_habilidades) > mejor_nuevas_habilidades:
                    mejor_nuevas_habilidades = len(nuevas_habilidades)
                    mejor_reactivo = reactivo
        
        if mejor_reactivo:
            genes_seleccionados.append(mejor_reactivo)
            habilidades_cubiertas.update(reactivos[mejor_reactivo])
        else:
            # Si no hay mejores opciones, seleccionar aleatoriamente
            restantes = [r for r in reactivos_disponibles if r not in genes_seleccionados]
            if restantes:
                genes_seleccionados.append(sample(restantes, 1)[0])
    
    return genes_seleccionados

def ejecutar_algoritmo_completo():
    """
    Ejecuta el algoritmo genético completo con validación
    """
    print("="*70)
    print("ALGORITMO GENÉTICO PARA OPTIMIZACIÓN DE REACTIVOS EDUCATIVOS")
    print("="*70)
    
    # Parámetros del sistema
    K = 3  # Número de reactivos por individuo
    tamaño_poblacion = 12
    generaciones = 5
    mutation_rate = 0.15
    selection_pressure = 0.6
    
    print(f"\nParámetros del sistema:")
    print(f"  K (reactivos por individuo): {K}")
    print(f"  Tamaño de población: {tamaño_poblacion}")
    print(f"  Generaciones: {generaciones}")
    print(f"  Tasa de mutación: {mutation_rate}")
    print(f"  Presión de selección: {selection_pressure}")
    
    # Mostrar estado inicial del sistema
    print(f"\n=== ESTADO INICIAL DEL SISTEMA ===")
    print("Matriz MRH (Reactivos vs Habilidades):")
    mostrar_tabla_MRH()
    
    print(f"\nHabilidades y calificaciones actuales:")
    habilidades_aprobadas = 0
    for habilidad, calificacion in sorted(hab.items()):
        estado = "✓ Aprobada" if calificacion >= 0.7 else "✗ No aprobada"
        print(f"  {habilidad}: {calificacion:.1f} - {estado}")
        if calificacion >= 0.7:
            habilidades_aprobadas += 1
    
    print(f"\nResumen: {habilidades_aprobadas}/{len(hab)} habilidades aprobadas")
    
    # Crear población inicial diversa
    print(f"\n=== CREANDO POBLACIÓN INICIAL ===")
    poblacion_inicial = crear_poblacion_diversa(tamaño_poblacion, K)
    
    print(f"Población inicial creada con {len(poblacion_inicial)} individuos:")
    for i, individuo in enumerate(poblacion_inicial):
        print(f"  {i+1:2d}. {individuo}")
    
    # Ejecutar algoritmo genético
    print(f"\n=== EJECUTANDO ALGORITMO GENÉTICO ===")
    ambiente = Environment(
        poblacion=poblacion_inicial,
        generations=generaciones,
        K=K,
        mutation_rate=mutation_rate,
        selection_pressure=selection_pressure
    )
    
    ambiente.start()
    
    # Análisis de resultados
    print(f"\n=== ANÁLISIS DE RESULTADOS ===")
    mejor_individuo = ambiente.get_best_individual()
    mejor_historico = ambiente.get_best_historical()
    
    print(f"Mejor individuo final:")
    print(f"  Genes: {mejor_individuo.gens}")
    print(f"  Fitness: {mejor_individuo.fitness:.4f}")
    print(f"  Métricas:")
    print(f"    OBJ1 (habilidades no aprobadas): {mejor_individuo.metrica_1:.3f}")
    print(f"    OBJ2 (reactivos ya realizados): {mejor_individuo.metrica_2}")
    print(f"    OBJ3 (reactivos con habs. aprobadas): {mejor_individuo.metrica_3}")
    print(f"    OBJ4 (habilidades involucradas): {mejor_individuo.metrica_4}")
    
    # Aplicar estrategia de validación
    print(f"\n=== ESTRATEGIA DE VALIDACIÓN ===")
    print("Simulando que los reactivos seleccionados se resuelven correctamente...")
    
    resultado_validacion = validar_estrategia(mejor_individuo, mostrar_detalle=True)
    
    # Validar toda la población
    print(f"\n=== VALIDACIÓN DE POBLACIÓN FINAL ===")
    resultados_poblacion = validar_poblacion(ambiente, mostrar_mejores=3)
    
    # Mostrar reporte completo
    mostrar_reporte_validacion(resultados_poblacion, mostrar_analisis=True)
    
    # Conclusiones
    print(f"\n=== CONCLUSIONES ===")
    mejor_validado = max(resultados_poblacion, key=lambda x: x["despues"]["fitness"])
    
    print(f"✓ Algoritmo genético ejecutado exitosamente")
    print(f"✓ Mejor solución encontrada: {mejor_validado['individuo'].gens}")
    print(f"✓ Fitness actual: {mejor_validado['antes']['fitness']:.4f}")
    print(f"✓ Fitness potencial: {mejor_validado['despues']['fitness']:.4f}")
    print(f"✓ Mejora esperada: {mejor_validado['mejora_fitness']:+.4f}")
    print(f"✓ Reactivos que se aprobarían: {mejor_validado['mejora_reactivos']}")
    print(f"✓ Habilidades que se aprobarían: {mejor_validado['mejora_habilidades']}")
    
    return ambiente, resultados_poblacion

def comparar_estrategias():
    """
    Compara diferentes estrategias de inicialización
    """
    print(f"\n=== COMPARACIÓN DE ESTRATEGIAS DE INICIALIZACIÓN ===")
    
    estrategias = {
        "Aleatoria": crear_poblacion_inicial,
        "Diversa": crear_poblacion_diversa
    }
    
    resultados_estrategias = {}
    
    for nombre, funcion_creacion in estrategias.items():
        print(f"\n--- Probando estrategia: {nombre} ---")
        
        poblacion = funcion_creacion(10, 3)
        ambiente = Environment(poblacion, generations=3)
        ambiente.start()
        
        mejor = ambiente.get_best_individual()
        promedio = sum(ind.fitness for ind in ambiente.poblacion) / len(ambiente.poblacion)
        
        resultados_estrategias[nombre] = {
            "mejor_fitness": mejor.fitness,
            "promedio_fitness": promedio,
            "mejor_genes": mejor.gens
        }
        
        print(f"Mejor fitness: {mejor.fitness:.4f}")
        print(f"Promedio fitness: {promedio:.4f}")
        print(f"Mejor genes: {mejor.gens}")
    
    print(f"\n--- Resumen de estrategias ---")
    for nombre, resultado in resultados_estrategias.items():
        print(f"{nombre}: Mejor={resultado['mejor_fitness']:.4f}, "
              f"Promedio={resultado['promedio_fitness']:.4f}")

def main():
    """Función principal"""
    # Fijar semilla para reproducibilidad (opcional)
    seed(42)
    
    # Ejecutar algoritmo completo
    ambiente, resultados = ejecutar_algoritmo_completo()
    
    # Comparar estrategias (opcional)
    # comparar_estrategias()
    
    print(f"\n{'='*70}")
    print("ALGORITMO GENÉTICO COMPLETADO EXITOSAMENTE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()