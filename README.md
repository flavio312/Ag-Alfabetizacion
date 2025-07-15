# Algoritmo Genético para Optimización de Reactivos Educativos

Este proyecto implementa un algoritmo genético para optimizar la selección de reactivos educativos basado en las habilidades del estudiante y métricas de aprendizaje específicas.

## 📁 Estructura del Proyecto

```
src/
├── models/
│   ├── individual/
│   │   └── individual.py          # Definición del individuo
│   └── environment/
│       └── environment.py         # Motor del algoritmo genético
├── test/
│   ├── main_model_SQL.py          # Datos del sistema y matrices
│   ├── validation.py              # Sistema de validación
│   └── test_validation.py         # Pruebas de validación
├── main_algorithm.py              # Algoritmo principal
└── ejemplo_uso.py                 # Ejemplos de uso
```

## 🧬 Componentes del Algoritmo Genético

### 1. **INDIVIDUO** (`src/models/individual/individual.py`)

#### Representación
- **Genes**: Lista de reactivos educativos (ej: `["R1", "R9", "R15"]`)
- **Cromosoma**: Secuencia ordenada de K reactivos alcanzables
- **Fenotipo**: Conjunto de habilidades involucradas en los reactivos

#### Función de Fitness
```python
fitness = [(1 + OBJ1) × (1 + OBJ4)] / [(1 + OBJ2) × (1 + OBJ3)]
```

**Métricas (Objetivos):**
- **OBJ1** (MAX): Ratio de habilidades no aprobadas involucradas
- **OBJ2** (MIN): Suma de veces que se han realizado los reactivos
- **OBJ3** (MIN): Cantidad de reactivos con habilidades ya aprobadas  
- **OBJ4** (MAX): Cantidad total de habilidades únicas involucradas

#### Ubicación del código:
```python
def _calculate_fitness(self):
    """Calcula el fitness según la fórmula especificada"""
    numerador = (1 + self.metrica_1) * (1 + self.metrica_4)
    denominador = (1 + self.metrica_2) * (1 + self.metrica_3)
    self.fitness = numerador / denominador
```

---

### 2. **EMPAREJAMIENTO/SELECCIÓN** (`src/models/environment/environment.py`)

#### Método: Selección por Torneo
- **Ubicación**: Método `select_pair()` (líneas 45-62)
- **Algoritmo**: 
  1. Selecciona candidatos aleatorios (tamaño del torneo = población × presión_selección)
  2. Elige el mejor fitness de cada torneo
  3. Garantiza que los padres sean diferentes

```python
def select_pair(self) -> Tuple[Individual, Individual]:
    """Selecciona un par de padres usando selección por torneo"""
    tournament_size = max(2, int(len(self.poblacion) * self.selection_pressure))
    
    # Torneo para padre 1
    candidates1 = sample(self.poblacion, min(tournament_size, len(self.poblacion)))
    padre = max(candidates1, key=lambda x: x.fitness)
    
    # Torneo para padre 2 (asegurar que sea diferente)
    candidates2 = sample(self.poblacion, min(tournament_size, len(self.poblacion)))
    madre = max(candidates2, key=lambda x: x.fitness)
```

#### Parámetros:
- `selection_pressure`: Controla intensidad de selección (0.0-1.0)
- Valores altos → más elitista, valores bajos → más diverso

---

### 3. **CRUZA/CROSSOVER** (`src/models/environment/environment.py`)

#### Método: Cruza de un punto con corrección de duplicados
- **Ubicación**: Método `crosses()` (líneas 64-85)
- **Ubicación corrección**: Método `_corregir_duplicados()` (líneas 87-120)

#### Algoritmo:
1. **Cruza básica**: Selecciona punto aleatorio y intercambia segmentos
2. **Corrección de duplicados** (según Estrategia 1 del documento):
   - Identifica reactivos faltantes: `FH = reactivos(padres) - reactivos(hijo)`
   - Encuentra duplicados en el hijo
   - Reemplaza duplicados con reactivos faltantes aleatoriamente

```python
def crosses(self):
    """Realiza cruza con corrección de duplicados"""
    for _ in range(len(self.poblacion)):
        padre, madre = self.select_pair()
        punto_cruza = randint(1, len(padre.gens) - 1)
        
        # Crear hijos iniciales
        hijo1_genes = padre.gens[:punto_cruza] + madre.gens[punto_cruza:]
        hijo2_genes = madre.gens[:punto_cruza] + padre.gens[punto_cruza:]
        
        # Corregir duplicados
        hijo1_genes = self._corregir_duplicados(hijo1_genes, padre.gens + madre.gens)
        hijo2_genes = self._corregir_duplicados(hijo2_genes, padre.gens + madre.gens)
```

#### Ejemplo de corrección:
```
Padre 1: [R1, R2, R3]     Madre: [R4, R5, R6]
Hijo inicial: [R1, R2, R5, R6]  ← Faltan R3, R4; sobra ninguno
Resultado: [R1, R2, R5, R6] ← Ya está correcto

Padre 1: [R1, R2, R3]     Madre: [R1, R5, R6]  
Hijo inicial: [R1, R2, R1, R6]  ← Falta R3, R5; sobra R1
Hijo corregido: [R1, R2, R3, R6]  ← Se reemplaza un R1 por R3
```

---

### 4. **MUTACIÓN** (`src/models/environment/environment.py`)

#### Método: Mutación por reemplazo de gen
- **Ubicación**: Método `mutate()` (líneas 122-135)
- **Probabilidad**: Controlada por `mutation_rate`

#### Algoritmo:
1. Para cada individuo, aplicar mutación según probabilidad
2. Seleccionar posición aleatoria en el cromosoma  
3. Reemplazar con reactivo no presente en el individuo
4. Recalcular fitness del individuo modificado

```python
def mutate(self):
    """Aplica mutación a algunos individuos"""
    for individuo in self.poblacion:
        if random() < self.mutation_rate:
            # Seleccionar posición aleatoria
            pos = randint(0, len(individuo.gens) - 1)
            
            # Nuevo reactivo que no esté ya presente
            reactivos_disponibles = [r for r in self.all_reactivos 
                                   if r not in individuo.gens]
            if reactivos_disponibles:
                nuevo_reactivo = choice(reactivos_disponibles)
                individuo.gens[pos] = nuevo_reactivo
                individuo.update_individual()  # Recalcular fitness
```

---

### 5. **SELECCIÓN DE SUPERVIVIENTES/PODA** (`src/models/environment/environment.py`)

#### Método: Selección elitista con diversidad
- **Ubicación**: Método `selection()` (líneas 137-156)
- **Estrategia**: Combina elitismo con diversidad

#### Algoritmo:
1. **Ordenar** población por fitness (descendente)
2. **Élite**: Mantener 50% de mejores individuos
3. **Diversidad**: Agregar 25% aleatorios del resto
4. **Resultado**: Nueva población con 75% del tamaño original

```python
def selection(self):
    """Selecciona supervivientes para la siguiente generación"""
    # Ordenar por fitness (descendente)
    self.poblacion.sort(key=lambda x: x.fitness, reverse=True)
    
    # Élite (mejores individuos)
    tamaño_elite = len(self.poblacion) // 2
    elite = self.poblacion[:tamaño_elite]
    
    # Algunos aleatorios para diversidad
    tamaño_aleatorio = len(self.poblacion) // 4  
    resto = self.poblacion[tamaño_elite:]
    if resto and tamaño_aleatorio > 0:
        aleatorios = sample(resto, min(tamaño_aleatorio, len(resto)))
    else:
        aleatorios = []
    
    # Nueva población = élite + aleatorios
    self.poblacion = elite + aleatorios
```

#### Ventajas:
- **Elitismo**: Preserva mejores soluciones
- **Diversidad**: Evita convergencia prematura
- **Reducción gradual**: Intensifica búsqueda con el tiempo

---

### 6. **BUCLE PRINCIPAL** (`src/models/environment/environment.py`)

#### Ubicación: Método `start()` (líneas 26-44)

#### Secuencia por generación:
```python
def start(self):
    for generation in range(self.generations):
        # 1. Guardar mejor individuo histórico
        mejor_actual = max(self.poblacion, key=lambda x: x.fitness)
        self.best_individuals_history.append(mejor_actual.copy())
        
        # 2. Mostrar estadísticas
        self._show_generation_stats(generation + 1)
        
        # 3. CRUZA: Generar descendencia
        self.crosses()
        
        # 4. MUTACIÓN: Aplicar mutaciones
        self.mutate()
        
        # 5. SELECCIÓN: Elegir supervivientes
        self.selection()
```

---

## 🚀 Uso del Sistema

### Ejecución Básica:
```python
from src.main_algorithm import ejecutar_algoritmo_completo

# Ejecutar con parámetros por defecto
ambiente, resultados = ejecutar_algoritmo_completo()
```

### Ejecución Personalizada:
```python
from src.models.environment.environment import Environment
from src.main_algorithm import crear_poblacion_diversa

# Crear población
poblacion = crear_poblacion_diversa(tamaño_poblacion=15, K=3)

# Configurar ambiente
ambiente = Environment(
    poblacion=poblacion,
    generations=10,           # Más generaciones
    K=3,                     # 3 reactivos por individuo
    mutation_rate=0.15,      # 15% de mutación
    selection_pressure=0.7   # Alta presión de selección
)

# Ejecutar
ambiente.start()

# Obtener mejor resultado
mejor = ambiente.get_best_individual()
print(f"Mejor solución: {mejor.gens}")
print(f"Fitness: {mejor.fitness:.4f}")
```

---

## 📊 Sistema de Validación

### Ubicación: `src/test/validation.py`

#### Estrategia de Validación:
1. **Simular éxito**: Asumir que reactivos seleccionados se resuelven correctamente
2. **Actualizar habilidades**: Colocar calificación 0.7 a habilidades < 0.7
3. **Recalcular métricas**: Evaluar impacto en fitness y objetivos
4. **MRH_temp**: Crear matriz temporal con calificaciones simuladas

#### Uso:
```python
from src.test.validation import validar_estrategia

resultado = validar_estrategia(individuo, mostrar_detalle=True)
print(f"Mejora esperada: {resultado['mejora_fitness']:+.4f}")
```

---

## ⚙️ Parámetros del Sistema

| Parámetro | Rango | Descripción | Valor Recomendado |
|-----------|-------|-------------|-------------------|
| `K` | 1-10 | Reactivos por individuo | 3 |
| `tamaño_poblacion` | 10-50 | Individuos en población | 12-20 |
| `generations` | 3-20 | Número de generaciones | 5-10 |
| `mutation_rate` | 0.05-0.25 | Probabilidad de mutación | 0.10-0.15 |
| `selection_pressure` | 0.3-0.9 | Intensidad de selección | 0.5-0.7 |

---

## 📈 Métricas y Objetivos

### Objetivos del Sistema:
1. **OBJ1** (MAX): Priorizar habilidades no aprobadas (especialmente las más bajas)
2. **OBJ2** (MIN): Evitar reactivos ya realizados múltiples veces  
3. **OBJ3** (MIN): Reducir uso de reactivos con habilidades ya dominadas
4. **OBJ4** (MAX): Maximizar cobertura de habilidades diferentes

### Interpretación del Fitness:
- **Alto** (>2.0): Excelente selección de reactivos
- **Medio** (1.0-2.0): Buena selección
- **Bajo** (<1.0): Selección sub-óptima

---

## 🔧 Extensiones Posibles

### 1. Operadores Genéticos Adicionales:
- Cruza uniforme
- Mutación por intercambio
- Cruza multipunto

### 2. Estrategias de Selección:
- Selección proporcional al fitness
- Selección por ranking
- Selección estocástica universal

### 3. Población Adaptativa:
- Tamaño de población variable
- Múltiples subpoblaciones (islas)
- Migración entre poblaciones

---

## 📝 Logging y Debugging

### Activar logs detallados:
```python
# En environment.py, cambiar show_generation_stats para más detalle
def _show_generation_stats(self, generation: int):
    # Agregar logs adicionales según necesidad
    pass
```

### Ver evolución:
```python
# Acceder al historial de mejores individuos
for i, individuo in enumerate(ambiente.best_individuals_history):
    print(f"Gen {i+1}: {individuo.fitness:.4f} - {individuo.gens}")
```

## 🎯 Casos de Uso Típicos

1. **Educación personalizada**: Seleccionar ejercicios óptimos para cada estudiante
2. **Remedial académico**: Enfocar en habilidades con mayor deficiencia  
3. **Evaluación adaptativa**: Maximizar información diagnóstica
4. **Planificación curricular**: Optimizar secuencias de aprendizaje