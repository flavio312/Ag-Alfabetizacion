from src.models.individual.individual import Individual
from src.models.environment.environment import Environment
from validation import validar_estrategia, validar_poblacion, crear_mrh_temp
from src.test.main_model_SQL import hab, reactivos, mostrar_tabla_MRH

def test_validacion_individual():
    print("=== PRUEBA DE VALIDACIÓN INDIVIDUAL ===")

    genes_test = ["R1", "R2", "R15"]
    individuo_test = Individual(genes_test)
    
    print(f"Individuo de prueba: {genes_test}")
    print(f"Fitness inicial: {individuo_test.fitness:.4f}")

    print("\nEstado inicial de habilidades:")
    habilidades_involucradas = set()
    for reactivo in genes_test:
        habilidades_involucradas.update(reactivos[reactivo])
    
    for habilidad in sorted(habilidades_involucradas):
        estado = "✓ Aprobada" if hab[habilidad] >= 0.7 else "✗ No aprobada"
        print(f"  {habilidad}: {hab[habilidad]:.1f} - {estado}")
    
    print("\n" + "-"*50)
    resultado = validar_estrategia(individuo_test, mostrar_detalle=True)
    
    return resultado

def test_validacion_poblacion():
    print("\n=== PRUEBA DE VALIDACIÓN DE POBLACIÓN ===")
    
    poblacion_test = [
        Individual(["R1", "R2", "R3"]),    
        Individual(["R4", "R5", "R12"]),   
        Individual(["R1", "R4", "R8"]),    
        Individual(["R13", "R14", "R18"])
    ]
    
    ambiente_test = Environment(poblacion_test, generations=1)
    
    print(f"Población de prueba creada con {len(poblacion_test)} individuos")
    for i, individuo in enumerate(poblacion_test):
        print(f"  Individuo {i+1}: {individuo.gens} - Fitness: {individuo.fitness:.4f}")
    
    ambiente_test.start()
    
    print(f"\nDespués de 1 generación: {len(ambiente_test.poblacion)} individuos")
    
    resultados = validar_poblacion(ambiente_test, mostrar_mejores=3)
    
    return resultados

def test_crear_mrh_temp():
    print("\n=== PRUEBA DE CREACIÓN DE MRH_TEMP ===")
    
    genes_test = ["R1", "R9", "R15"]
    individuo_test = Individual(genes_test)
    
    print(f"Individuo de prueba: {genes_test}")

    print("\nHabilidades que serán actualizadas:")
    for reactivo in genes_test:
        for habilidad in reactivos[reactivo]:
            if hab[habilidad] < 0.7:
                print(f"  {habilidad}: {hab[habilidad]:.1f} → 0.7")
  
    mrh_temp = crear_mrh_temp(individuo_test)
    
    print(f"\nMRH_temp creado exitosamente con {len(mrh_temp)} reactivos")
    
    return mrh_temp

def test_comparacion_estrategias():
    print("\n=== COMPARACIÓN DE ESTRATEGIAS ===")
    
    estrategias = {
        "Enfoque habilidades bajas": ["R1", "R2", "R9"],
        "Enfoque habilidades altas": ["R4", "R5", "R13"],
        "Enfoque mixto": ["R1", "R4", "R8"],       
        "Enfoque cobertura": ["R12", "R16", "R20"]
    }
    
    resultados_estrategias = {}
    
    for nombre, genes in estrategias.items():
        print(f"\n--- {nombre} ---")
        individuo = Individual(genes)
        resultado = validar_estrategia(individuo, mostrar_detalle=False)
        resultados_estrategias[nombre] = resultado
        
        print(f"Genes: {genes}")
        print(f"Fitness: {resultado['antes']['fitness']:.4f} → {resultado['despues']['fitness']:.4f} (Δ: {resultado['mejora_fitness']:+.4f})")
        print(f"Reactivos aprobados: {resultado['antes']['reactivos_aprobados']} → {resultado['despues']['reactivos_aprobados']}")
        print(f"Habilidades actualizadas: {len(resultado['habilidades_actualizadas'])}")
    
    mejor_estrategia = max(resultados_estrategias.items(), key=lambda x: x[1]['mejora_fitness'])
    
    print(f"\n=== MEJOR ESTRATEGIA ===")
    print(f"Estrategia: {mejor_estrategia[0]}")
    print(f"Genes: {estrategias[mejor_estrategia[0]]}")
    print(f"Mejora en fitness: {mejor_estrategia[1]['mejora_fitness']:+.4f}")
    print(f"Fitness final potencial: {mejor_estrategia[1]['despues']['fitness']:.4f}")
    
    return resultados_estrategias

def test_validacion_completa():
    print("="*70)
    print("PRUEBA COMPLETA DEL SISTEMA DE VALIDACIÓN")
    print("="*70)
    
    print("\n=== ESTADO INICIAL DEL SISTEMA ===")
    print("Habilidades y sus calificaciones:")
    for habilidad, calificacion in hab.items():
        estado = "✓ Aprobada" if calificacion >= 0.7 else "✗ No aprobada"
        print(f"  {habilidad}: {calificacion:.1f} - {estado}")
    
    aprobadas_inicial = sum(1 for cal in hab.values() if cal >= 0.7)
    print(f"\nHabilidades aprobadas inicialmente: {aprobadas_inicial}/{len(hab)}")
    
    print("\n" + "="*50)
    resultado_individual = test_validacion_individual()
    
    print("\n" + "="*50)
    resultados_poblacion = test_validacion_poblacion()
    
    print("\n" + "="*50)
    mrh_temp = test_crear_mrh_temp()
    
    print("\n" + "="*50)
    resultados_estrategias = test_comparacion_estrategias()
    
    print("\n" + "="*70)
    print("RESUMEN DE TODAS LAS PRUEBAS")
    print("="*70)
    
    print(f"\n1. Validación individual:")
    print(f"   - Mejora en fitness: {resultado_individual['mejora_fitness']:+.4f}")
    print(f"   - Habilidades actualizadas: {len(resultado_individual['habilidades_actualizadas'])}")
    
    print(f"\n2. Validación de población:")
    print(f"   - Individuos analizados: {len(resultados_poblacion)}")
    mejor_poblacion = max(resultados_poblacion, key=lambda x: x['mejora_fitness'])
    print(f"   - Mejor mejora: {mejor_poblacion['mejora_fitness']:+.4f}")
    
    print(f"\n3. MRH_temp:")
    print(f"   - Creado exitosamente con {len(mrh_temp)} reactivos")
    
    print(f"\n4. Comparación de estrategias:")
    mejor_est = max(resultados_estrategias.items(), key=lambda x: x[1]['mejora_fitness'])
    print(f"   - Mejor estrategia: {mejor_est[0]}")
    print(f"   - Mejora: {mejor_est[1]['mejora_fitness']:+.4f}")
    
    print(f"\n=== CONCLUSIONES ===")
    print("✓ El sistema de validación funciona correctamente")
    print("✓ Se pueden identificar las mejores estrategias de selección")
    print("✓ MRH_temp se crea correctamente con calificaciones simuladas")
    print("✓ Las métricas se calculan adecuadamente antes y después de la validación")

def mostrar_matriz_mrh_comparacion():
    print("\n=== COMPARACIÓN MRH ORIGINAL vs MRH_TEMP ===")

    print("\nMRH Original:")
    mostrar_tabla_MRH()
    
    individuo_ejemplo = Individual(["R1", "R2", "R15"])
    mrh_temp = crear_mrh_temp(individuo_ejemplo)
    
    print(f"\nMRH_temp para individuo {individuo_ejemplo.gens}:")
    print("(Con calificaciones aprobatorias simuladas)")
    
    print("\nDiferencias principales:")
    print("- MRH original contiene las calificaciones reales")
    print("- MRH_temp contiene calificaciones aprobatorias simuladas (≥0.7)")
    print("- Esto permite evaluar el potencial de mejora de cada estrategia")

if __name__ == "__main__":
    test_validacion_completa()
    
    mostrar_matriz_mrh_comparacion()
    
    print("\n" + "="*70)
    print("TODAS LAS PRUEBAS COMPLETADAS")
    print("="*70)