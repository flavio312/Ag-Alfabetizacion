# Algoritmo Genético para Selección de Reactivos Educativos

## 📋 Descripción General

Este proyecto implementa un **algoritmo genético especializado** para optimizar la selección de reactivos educativos basándose en las habilidades del estudiante y métricas de aprendizaje específicas. El sistema está diseñado para maximizar el aprendizaje del estudiante priorizando habilidades deficientes mientras minimiza la redundancia.

## 🎯 Objetivos del Sistema

El algoritmo optimiza simultáneamente cuatro objetivos clave:

- **OBJ1 (Maximizar)**: Uso de reactivos que involucran habilidades no aprobadas (priorizando las más bajas)
- **OBJ2 (Minimizar)**: Uso de reactivos ya realizados anteriormente
- **OBJ3 (Minimizar)**: Uso de reactivos con habilidades ya aprobadas
- **OBJ4 (Maximizar)**: Cantidad de habilidades únicas involucradas

### Función de Fitness

```
Fitness = [(1 + OBJ1) × (1 + OBJ4)] / [(1 + OBJ2) × (1 + OBJ3)]
```

## 🏗️ Arquitectura del Sistema

### Componentes Principales

```
src/
├── models/
│   ├── individual/          # Definición de individuos del AG
│   └── environment/         # Motor del algoritmo genético
├── visualization/           # Sistema de gráficas y reportes
├── validation/              # Simulación y validación de estrategias
└── examples/               # Ejemplos de uso y datos de prueba
```

### Clases Fundamentales

#### 1. `Reactivo`
Representa un ejercicio educativo con sus habilidades asociadas.

```python
@dataclass
class Reactivo:
    id: str                                    # Identificador único
    habilidades: List[str]                     # Lista de habilidades involucradas
    peso_habilidades: Dict[str, float]         # Pesos normalizados (suman 1.0)
```

**Características clave:**
- ✅ Soporte para 1 a n habilidades por reactivo
- ✅ Normalización automática de pesos (garantiza suma = 1.0)
- ✅ Distribución equitativa automática si no se especifican pesos

#### 2. `Habilidad`
Representa una competencia educativa con su estado actual.

```python
@dataclass
class Habilidad:
    id: str                                    # Identificador único
    calificacion: float = 0.0                  # Calificación actual [0.0-1.0]
    aprobada: bool = False                     # Estado de aprobación
    umbral_aprobacion: float = 0.7             # Umbral para aprobar
```

#### 3. `Individual`
Cromosoma del algoritmo genético que representa una solución candidata.

```python
class Individual:
    genes: List[str]                           # Lista de K reactivos seleccionados
    fitness: float                             # Valor de aptitud calculado
    metricas: Dict                             # Valores de OBJ1, OBJ2, OBJ3, OBJ4
```

## 🧬 Algoritmo Genético

### Representación Genética

- **Gen**: ID de un reactivo educativo (ej: "R1", "R9", "R15", hasta K reactivos)
- **Cromosoma**: Secuencia ordenada de K reactivos alcanzables
- **Fenotipo**: Conjunto de habilidades involucradas en los reactivos

### Operadores Genéticos

#### 1. **Selección por Torneo**
```python
def seleccionar_padres(self) -> Tuple[Individual, Individual]:
    tamaño_torneo = max(2, int(len(self.poblacion) * self.presion_seleccion))
    # Selecciona el mejor de cada torneo
```

#### 2. **Cruza con Corrección de Duplicados**
Implementa la **Estrategia 1** especificada en el documento:

```python
def cruzar(self, padre1: Individual, padre2: Individual):
    # 1. Cruza en punto aleatorio
    punto_cruza = random.randint(1, len(padre1.genes) - 1)
    hijo1_genes = padre1.genes[:punto_cruza] + padre2.genes[punto_cruza:]
    
    # 2. Corregir duplicados
    hijo1_genes = self._corregir_duplicados(hijo1_genes, todos_los_genes)
```

**Proceso de corrección:**
1. Identificar reactivos faltantes: `FH = reactivos(padres) - reactivos(hijo)`
2. Buscar duplicados en el hijo
3. Reemplazar duplicados con reactivos faltantes aleatorios

#### 3. **Mutación Inteligente**
```python
def mutar(self, individuo: Individual):
    if random.random() < self.tasa_mutacion:
        # Seleccionar posición aleatoria
        pos = random.randint(0, len(individuo.genes) - 1)
        # Reemplazar con reactivo no presente
        nuevo_reactivo = choice(reactivos_no_presentes)
```

#### 4. **Selección de Supervivientes**
Estrategia híbrida que combina:
- **Elitismo** (50%): Preserva mejores soluciones
- **Diversidad** (25%): Mantiene variabilidad genética
- **Reducción gradual**: Intensifica búsqueda con el tiempo

## 📊 Sistema de Visualización

### Gráficas Implementadas

#### 1. **Evolución del Fitness**
```python
def graficar_evolucion_fitness(self):
    # Muestra progreso del mejor fitness por generación
    # Incluye zoom a últimas 10 generaciones
```

#### 2. **Métricas de Objetivos**
```python
def graficar_metricas_objetivos(self):
    # Evolución de OBJ1, OBJ2, OBJ3, OBJ4
    # 4 subgráficas con interpretación clara
```

#### 3. **Estado de Habilidades**
```python
def graficar_estado_habilidades(self):
    # Barras: Calificaciones por habilidad
    # Pastel: Distribución aprobadas/no aprobadas
```

#### 4. **Análisis de Reactivos Seleccionados**
```python
def graficar_analisis_reactivos_seleccionados(self):
    # Habilidades por reactivo
    # Distribución de pesos
    # Mapa de calor reactivos vs habilidades
    # Métricas del mejor individuo
```

## 🔬 Sistema de Validación

### Simulación de Mejora Esperada

El sistema puede predecir el impacto de resolver correctamente los reactivos seleccionados:

```python
def simular_mejora(self, individuo: Individual) -> Dict:
    # 1. Crear copia temporal de habilidades
    # 2. Simular mejora proporcional a pesos
    # 3. Recalcular métricas
    # 4. Retornar comparación antes/después
```

**Salida de la simulación:**
- Fitness antes vs después
- Habilidades aprobadas antes vs después
- Número de nuevas habilidades que se aprobarían
- Mejora esperada en fitness

## 🚀 Uso del Sistema

### Instalación de Dependencias

```bash
pip install numpy matplotlib pandas seaborn dataclasses copy random typing
```

### Uso Básico

```python
# 1. Crear datos del problema
reactivos_data, habilidades_data, conteo_reactivos, reactivos_alcanzables = crear_datos_ejemplo()

# 2. Configurar algoritmo genético
ag = AlgoritmoGenetico(
    reactivos_alcanzables=reactivos_alcanzables,
    reactivos_data=reactivos_data,
    habilidades_data=habilidades_data,
    conteo_reactivos=conteo_reactivos,
    K=3,                    # Seleccionar 3 reactivos
    tamaño_poblacion=20,    # 20 individuos
    generaciones=50,        # 50 iteraciones
    tasa_mutacion=0.15,     # 15% mutación
    presion_seleccion=0.7   # Alta presión
)

# 3. Ejecutar evolución
ag.evolucionar()

# 4. Generar visualizaciones
visualizador = VisualizadorResultados(ag)
visualizador.generar_reporte_completo()
```

### Configuración Avanzada

```python
# Obtener recomendaciones de parámetros
config = ConfiguradorParametros.recomendar_parametros(
    num_reactivos=10,
    num_habilidades=8,
    complejidad_problema="medio"  # "simple", "medio", "complejo"
)

# Aplicar configuración recomendada
ag = AlgoritmoGenetico(**config, ...)
```

## ⚙️ Parámetros Recomendados

| Parámetro | Rango | Descripción | Valor Recomendado |
|-----------|-------|-------------|-------------------|
| `K` | 1-10 | Reactivos por individuo | 3-5 |
| `tamaño_poblacion` | 10-50 | Individuos en población | 20-30 |
| `generaciones` | 10-100 | Número de iteraciones | 50 |
| `tasa_mutacion` | 0.05-0.25 | Probabilidad de mutación | 0.10-0.15 |
| `presion_seleccion` | 0.3-0.9 | Intensidad de selección | 0.6-0.7 |

### Interpretación de Fitness

- **Alto (>2.0)**: Excelente selección de reactivos
- **Medio (1.0-2.0)**: Buena selección, puede mejorarse
- **Bajo (<1.0)**: Selección sub-óptima, revisar parámetros

## 📈 Ejemplo de Datos

### Reactivos de Ejemplo
```python
reactivos_data = {
    'R1': Reactivo('R1', ['H1', 'H3', 'H5'], {'H1': 0.4, 'H3': 0.3, 'H5': 0.3}),
    'R2': Reactivo('R2', ['H3', 'H4'], {'H3': 0.6, 'H4': 0.4}),
    'R3': Reactivo('R3', ['H1', 'H2', 'H4', 'H6']),  # Pesos automáticos (0.25 c/u)
    'R4': Reactivo('R4', ['H2', 'H5']),               # Pesos automáticos (0.5 c/u)
    'R5': Reactivo('R5', ['H6']),                     # Una sola habilidad (1.0)
}
```

### Habilidades de Ejemplo
```python
habilidades_data = {
    'H1': Habilidad('H1', 0.9),   # Aprobada (90%)
    'H2': Habilidad('H2', 1.0),   # Aprobada (100%)
    'H3': Habilidad('H3', 0.5),   # No aprobada (50%) - Prioridad alta
    'H4': Habilidad('H4', 0.0),   # No aprobada (0%) - Prioridad muy alta
    'H5': Habilidad('H5', 0.0),   # No aprobada (0%) - Prioridad muy alta
    'H6': Habilidad('H6', 0.0),   # No aprobada (0%) - Prioridad muy alta
}
```

## 🔧 Validaciones Implementadas

### 1. **Restricción de Pesos**
```python
def validar_restricciones():
    for reactivo_id, reactivo in reactivos_data.items():
        suma_pesos = sum(reactivo.peso_habilidades.values())
        assert abs(suma_pesos - 1.0) < 1e-6, f"Los pesos del reactivo {reactivo_id} no suman 1"
```

### 2. **Corrección Automática**
- Los pesos se normalizan automáticamente al crear reactivos
- Se valida que no haya duplicados en los cromosomas
- Se verifica que todos los reactivos existan en el conjunto alcanzable

## 📋 Salida del Sistema

### Reporte Completo
```
MEJOR SOLUCIÓN ENCONTRADA:
Reactivos seleccionados: ['R3', 'R4', 'R5']
Fitness: 2.3456
Métricas:
  OBJ1: 0.857
  OBJ2: 1.000
  OBJ3: 0.000
  OBJ4: 5.000

SIMULACIÓN DE MEJORA:
Fitness actual: 2.3456
Fitness esperado: 3.1234
Mejora esperada: 0.7778
Habilidades aprobadas antes: 2
Habilidades aprobadas después: 4
Nuevas habilidades que se aprobarían: 2
```

## 🎨 Características Destacadas

### ✅ **Implementado**
- [x] Pesos de habilidades normalizados (suman 1.0)
- [x] Soporte para reactivos con 2-n habilidades
- [x] Gráficas completas de evaluación
- [x] Simulación de mejora esperada
- [x] Corrección automática de duplicados
- [x] Configuración flexible de parámetros
- [x] Validación de restricciones
- [x] Reporte completo con métricas

### 🚀 **Posibles Extensiones**
- [ ] Cruza uniforme y multipunto
- [ ] Múltiples subpoblaciones (algoritmo de islas)
- [ ] Optimización multiobjetivo (NSGA-II)
- [ ] Aprendizaje de parámetros dinámicos
- [ ] Integración con base de datos real
- [ ] API REST para uso remoto

## 📚 Referencias

- **Documento base**: Especificación del algoritmo genético para alfabetización
- **Estrategia de cruza**: Estrategia 1 para corrección de duplicados
- **Función de fitness**: Fórmula multiobjetivo especificada
- **Validación**: Sistema de simulación de mejora esperada

## 📞 Soporte

Para reportar problemas o sugerir mejoras, por favor:
1. Revisar la documentación completa
2. Verificar que los datos cumplan las restricciones
3. Probar con parámetros recomendados
4. Contactar al desarrollador con ejemplos específicos

---

**Nota**: Este algoritmo está específicamente diseñado para optimización educativa y puede requerir ajustes para otros dominios de aplicación.