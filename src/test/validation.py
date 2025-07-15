from typing import List, Dict, Any
from src.models.individual.individual import Individual
from src.models.environment.environment import Environment
from src.test.main_model_SQL import hab, reactivos, mostrar_tabla_de, MRH

def validar_estrategia(individual: Individual, mostrar_detalle: bool = True) -> Dict[str, Any]:
    hab_original = hab.copy()
    
    metricas_antes = {
        "reactivos_aprobados": _contar_reactivos_aprobados(individual.gens, hab_original),
        "habilidades_no_aprobadas": len(individual.habs_no_aprob),
        "habilidades_aprobadas": len(individual.habs_aprob),
        "fitness": individual.fitness,
        "metrica_1": individual.metrica_1,
        "metrica_2": individual.metrica_2,
        "metrica_3": individual.metrica_3,
        "metrica_4": individual.metrica_4
    }
    
    habilidades_actualizadas = []
    for reactivo in individual.gens:
        habilidades_reactivo = reactivos[reactivo]
        for habilidad in habilidades_reactivo:
            if hab[habilidad] < 0.7:
                habilidades_actualizadas.append({
                    'habilidad': habilidad,
                    'valor_anterior': hab[habilidad],
                    'valor_nuevo': 0.7,
                    'reactivo': reactivo
                })
                hab[habilidad] = 0.7

    individual_temp = Individual(individual.gens.copy())
    
    metricas_despues = {
        "reactivos_aprobados": _contar_reactivos_aprobados(individual_temp.gens, hab),
        "habilidades_no_aprobadas": len(individual_temp.habs_no_aprob),
        "habilidades_aprobadas": len(individual_temp.habs_aprob),
        "fitness": individual_temp.fitness,
        "metrica_1": individual_temp.metrica_1,
        "metrica_2": individual_temp.metrica_2,
        "metrica_3": individual_temp.metrica_3,
        "metrica_4": individual_temp.metrica_4
    }
    
    if mostrar_detalle:
        _mostrar_detalle_validacion(individual, metricas_antes, metricas_despues, habilidades_actualizadas)
    
    hab.update(hab_original)
    
    return {
        "antes": metricas_antes,
        "despues": metricas_despues,
        "mejora_reactivos": metricas_despues["reactivos_aprobados"] - metricas_antes["reactivos_aprobados"],
        "mejora_habilidades": metricas_despues["habilidades_aprobadas"] - metricas_antes["habilidades_aprobadas"],
        "mejora_fitness": metricas_despues["fitness"] - metricas_antes["fitness"],
        "habilidades_actualizadas": habilidades_actualizadas
    }


def validar_poblacion(environment: Environment, mostrar_mejores: int = 3) -> List[Dict[str, Any]]:
    resultados = []
    
    print("=== VALIDACIÓN DE POBLACIÓN ===")
    print(f"Validando {len(environment.poblacion)} individuos...\n")
    
    for i, individuo in enumerate(environment.poblacion):
        resultado = validar_estrategia(individuo, mostrar_detalle=False)
        resultado["individuo"] = individuo
        resultado["indice"] = i
        resultados.append(resultado)
    
    resultados.sort(key=lambda x: x["mejora_fitness"], reverse=True)
    
    print(f"--- TOP {mostrar_mejores} INDIVIDUOS CON MAYOR MEJORA ---")
    for i, resultado in enumerate(resultados[:mostrar_mejores]):
        print(f"\n{i+1}. Individuo {resultado['indice']}")
        print(f"   Genes: {resultado['individuo'].gens}")
        print(f"   Fitness: {resultado['antes']['fitness']:.4f} → {resultado['despues']['fitness']:.4f} (Δ: {resultado['mejora_fitness']:+.4f})")
        print(f"   Reactivos aprobados: {resultado['antes']['reactivos_aprobados']} → {resultado['despues']['reactivos_aprobados']}")
        print(f"   Habilidades aprobadas: {resultado['antes']['habilidades_aprobadas']} → {resultado['despues']['habilidades_aprobadas']}")
        print(f"   Habilidades actualizadas: {len(resultado['habilidades_actualizadas'])}")
    
    return resultados


def crear_mrh_temp(individual: Individual) -> Dict[str, List[float]]:
    mrh_temp = MRH.copy()
    
    hab_original = hab.copy()
    
    for reactivo in individual.gens:
        habilidades_reactivo = reactivos[reactivo]
        for habilidad in habilidades_reactivo:
            if hab[habilidad] < 0.7:
                hab[habilidad] = 0.7
    
    habilidades_lista = list(hab.keys())
    for reactivo_key, habilidades_reactivo in reactivos.items():
        mrh_temp[reactivo_key] = []
        for habilidad in habilidades_lista:
            if habilidad in habilidades_reactivo:
                mrh_temp[reactivo_key].append(hab[habilidad])
            else:
                mrh_temp[reactivo_key].append(0.0)
    
    hab.update(hab_original)
    
    return mrh_temp


def analizar_impacto_validacion(resultados: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not resultados:
        return {}
    
    mejoras_fitness = [r["mejora_fitness"] for r in resultados]
    mejoras_reactivos = [r["mejora_reactivos"] for r in resultados]
    mejoras_habilidades = [r["mejora_habilidades"] for r in resultados]
    
    analisis = {
        "fitness": {
            "promedio": sum(mejoras_fitness) / len(mejoras_fitness),
            "maximo": max(mejoras_fitness),
            "minimo": min(mejoras_fitness),
            "positivas": sum(1 for m in mejoras_fitness if m > 0),
            "negativas": sum(1 for m in mejoras_fitness if m < 0),
            "neutras": sum(1 for m in mejoras_fitness if m == 0)
        },
        "reactivos": {
            "promedio": sum(mejoras_reactivos) / len(mejoras_reactivos),
            "maximo": max(mejoras_reactivos),
            "minimo": min(mejoras_reactivos)
        },
        "habilidades": {
            "promedio": sum(mejoras_habilidades) / len(mejoras_habilidades),
            "maximo": max(mejoras_habilidades),
            "minimo": min(mejoras_habilidades)
        },
        "total_individuos": len(resultados)
    }
    
    return analisis

def mostrar_reporte_validacion(resultados: List[Dict[str, Any]], mostrar_analisis: bool = True):
    print("\n" + "="*60)
    print("REPORTE DE VALIDACIÓN COMPLETO")
    print("="*60)
    
    if mostrar_analisis:
        analisis = analizar_impacto_validacion(resultados)
        
        print("\n--- ANÁLISIS ESTADÍSTICO ---")
        print(f"Total de individuos analizados: {analisis['total_individuos']}")
        
        print(f"\nFitness:")
        print(f"  Mejora promedio: {analisis['fitness']['promedio']:+.4f}")
        print(f"  Mejora máxima: {analisis['fitness']['maximo']:+.4f}")
        print(f"  Mejora mínima: {analisis['fitness']['minimo']:+.4f}")
        print(f"  Individuos con mejora positiva: {analisis['fitness']['positivas']}")
        print(f"  Individuos con mejora negativa: {analisis['fitness']['negativas']}")
        print(f"  Individuos sin cambio: {analisis['fitness']['neutras']}")
        
        print(f"\nReactivos aprobados:")
        print(f"  Mejora promedio: {analisis['reactivos']['promedio']:+.2f}")
        print(f"  Mejora máxima: {analisis['reactivos']['maximo']:+.0f}")
        
        print(f"\nHabilidades aprobadas:")
        print(f"  Mejora promedio: {analisis['habilidades']['promedio']:+.2f}")
        print(f"  Mejora máxima: {analisis['habilidades']['maximo']:+.0f}")
    
    mejor_resultado = max(resultados, key=lambda x: x["despues"]["fitness"])
    print(f"\n--- MEJOR INDIVIDUO DESPUÉS DE VALIDACIÓN ---")
    print(f"Genes: {mejor_resultado['individuo'].gens}")
    print(f"Fitness final: {mejor_resultado['despues']['fitness']:.4f}")
    print(f"Reactivos aprobados: {mejor_resultado['despues']['reactivos_aprobados']}")
    print(f"Habilidades aprobadas: {mejor_resultado['despues']['habilidades_aprobadas']}")
    print(f"Habilidades actualizadas: {len(mejor_resultado['habilidades_actualizadas'])}")


def _contar_reactivos_aprobados(genes: List[str], hab_dict: Dict[str, float]) -> int:
    reactivos_aprobados = 0
    for reactivo in genes:
        habilidades_reactivo = reactivos[reactivo]
        if all(hab_dict[h] >= 0.7 for h in habilidades_reactivo):
            reactivos_aprobados += 1
    return reactivos_aprobados


def _mostrar_detalle_validacion(individual: Individual, metricas_antes: Dict,
                                metricas_despues: Dict, habilidades_actualizadas: List[Dict]):

    print("=== ESTRATEGIA DE VALIDACIÓN ===")
    print(f"Reactivos seleccionados: {individual.gens}")
    print("\n--- COMPARACIÓN ANTES/DESPUÉS ---")
    print(f"Reactivos aprobados: {metricas_antes['reactivos_aprobados']} → {metricas_despues['reactivos_aprobados']}")
    print(f"Habilidades no aprobadas: {metricas_antes['habilidades_no_aprobadas']} → {metricas_despues['habilidades_no_aprobadas']}")
    print(f"Habilidades aprobadas: {metricas_antes['habilidades_aprobadas']} → {metricas_despues['habilidades_aprobadas']}")
    print(f"Fitness: {metricas_antes['fitness']:.4f} → {metricas_despues['fitness']:.4f}")
    
    print(f"\n--- MÉTRICAS DETALLADAS ---")
    print(f"Métrica 1: {metricas_antes['metrica_1']:.4f} → {metricas_despues['metrica_1']:.4f}")
    print(f"Métrica 2: {metricas_antes['metrica_2']:.4f} → {metricas_despues['metrica_2']:.4f}")
    print(f"Métrica 3: {metricas_antes['metrica_3']:.4f} → {metricas_despues['metrica_3']:.4f}")
    print(f"Métrica 4: {metricas_antes['metrica_4']:.4f} → {metricas_despues['metrica_4']:.4f}")
    
    print("\n--- TABLA DE VALIDACIÓN ---")
    print(mostrar_tabla_de(individual.gens))
    
    if habilidades_actualizadas:
        print(f"\n--- HABILIDADES ACTUALIZADAS ---")
        for hab_info in habilidades_actualizadas:
            print(f"{hab_info['habilidad']}: {hab_info['valor_anterior']:.1f} → {hab_info['valor_nuevo']:.1f} (Reactivo: {hab_info['reactivo']})")
    else:
        print("\nNo se actualizaron habilidades (ya estaban aprobadas)")

if __name__ == "__main__":
    from random import sample
    
    genes_ejemplo = sample(list(reactivos.keys()), 3)
    individual_ejemplo = Individual(genes_ejemplo)
    
    print("=== EJEMPLO DE VALIDACIÓN ===")
    print(f"Individuo de ejemplo con genes: {genes_ejemplo}")
    
    resultado = validar_estrategia(individual_ejemplo)
    
    print(f"\nResultado de validación:")
    print(f"Mejora en fitness: {resultado['mejora_fitness']:+.4f}")
    print(f"Mejora en reactivos: {resultado['mejora_reactivos']:+.0f}")
    print(f"Mejora en habilidades: {resultado['mejora_habilidades']:+.0f}")