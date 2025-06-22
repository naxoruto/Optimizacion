# Traveling Thief Problem - Simulated Annealing Solver

Este proyecto resuelve el problema conocido como Traveling Thief Problem (TTP), que combina el problema del viajante (TSP) y el de la mochila (KP). Utiliza un enfoque de Simulated Annealing en paralelo para encontrar soluciones eficientes.

## Descripción

- Lee archivos `.ttp` con la definición del problema (ciudades, ítems, capacidades, etc).
- Ejecuta múltiples búsquedas en paralelo para encontrar la mejor solución.
- Permite visualizar la mejor ruta encontrada y desglosar el profit y la penalización por tiempo.

## Requisitos

- Python 3.7+
- numpy
- matplotlib
- numba

Instala las dependencias con:

```sh
pip install numpy matplotlib numba
```

## Uso

Ejecuta el script principal con los siguientes argumentos:

```sh
python final_ttp_solver_with_profit.py --file RUTA_AL_ARCHIVO_TTP [opciones]
```

### Argumentos principales

- `--file`: Ruta al archivo `.ttp` (obligatorio)
- `--parallel-runs`: Número de ejecuciones en paralelo (por defecto 8)
- `--fast-test`: Usa configuración ligera para pruebas rápidas
- `--show-profit`: Muestra el desglose de profit y penalización
- `--profile`: Mide el tiempo de ejecución
- `--plot`: Muestra una visualización de la mejor ruta

### Ejemplo

```sh
python final_ttp_solver_with_profit.py --file TTP3.ttp --show-profit --plot
```

## Archivos principales

- `final_ttp_solver_with_profit.py`: Script principal con toda la lógica del solver y visualización.
- Archivos `.ttp`: Instancias del problema a resolver.

## Créditos

Desarrollado por Ignacio y Rubén.
