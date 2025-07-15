# src/test/main_model_SQL.py
from tabulate import tabulate
#implementar que Al sumar las habilidades de los reactivos deben de dar 1
hab = {
    "H1": 0.9,
    "H2": 1.0,
    "H3": 0.5,
    "H4": 0.0,
    "H5": 0.0,
    "H6": 0.0,
    "H7": 0.3,
    "H8": 0.7,
    "H9": 0.2,
    "H10": 0.8
}

reactivos = {
    "R1": ["H1", "H3", "H5"],
    "R2": ["H4", "H7"],
    "R3": ["H6", "H8"],
    "R4": ["H2", "H9", "H10"],
    "R5": ["H1", "H8", "H9"],
    "R6": ["H4", "H10"],
    "R7": ["H2", "H3", "H7"],
    "R8": ["H1", "H5", "H10"],
    "R9": ["H5", "H7"],
    "R10": ["H2", "H6", "H9"],
    "R11": ["H3", "H5", "H8"],
    "R12": ["H1", "H2", "H4", "H10"],
    "R13": ["H2", "H8", "H9"],
    "R14": ["H1", "H2", "H7"],
    "R15": ["H3", "H4", "H10"],
    "R16": ["H1", "H4", "H6"],
    "R17": ["H2", "H5", "H9"],
    "R18": ["H1", "H7"],
    "R19": ["H4", "H8", "H10"],
    "R20": ["H3", "H6", "H9"]
}

reactivos_realizados = {
    "R1": 2,
    "R2": 3,
    "R3": 0,
    "R4": 4,
    "R5": 1,
    "R6": 5,
    "R7": 0,
    "R8": 3,
    "R9": 2,
    "R10": 4,
    "R11": 1,
    "R12": 5,
    "R13": 0,
    "R14": 3,
    "R15": 2,
    "R16": 4,
    "R17": 1,
    "R18": 5,
    "R19": 0,
    "R20": 3
}

MRH = {}
habilidades = list(hab.keys())

for r, hs in reactivos.items():
    MRH[r] = [1 if h in hs else 0 for h in habilidades]

def mostrar_tabla_de(lista_reactivos):
    habilidades_unicas = list(hab.keys())

    tabla = []
    for reactivo_key in lista_reactivos:
        if reactivo_key in reactivos:
            fila = []
            habilidades_en_reactivo = {h: hab[h] for h in reactivos[reactivo_key]}
            for habilidad in habilidades_unicas:
                valor = habilidades_en_reactivo.get(habilidad, -1)
                fila.append(valor)
            tabla.append([reactivo_key] + fila)

    headers = ["G"] + habilidades_unicas
    return tabulate(tabla, headers=headers, tablefmt="fancy_grid", floatfmt=".2f")

def mostrar_tabla_MRH():
    habilidades_unicas = list(hab.keys())
    tabla = []
    for reactivo, hs in reactivos.items():
        fila = [1 if h in hs else 0 for h in habilidades_unicas]
        tabla.append([reactivo] + fila)
    headers = ["Reactivo"] + habilidades_unicas
    print(tabulate(tabla, headers=headers, tablefmt="fancy_grid"))