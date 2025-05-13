# TP3 - SIA - Grupo 1

## Requisitos

- Python 3.8 o superior
- uv: gestor de entornos y dependencias

### Instalación de uv:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### Sincronización de dependencias:

Este proyecto incluye un archivo requirements.txt. Para instalar todo con uv, ejecutar:

```bash
uv venv .venv  
uv sync
```

Esto crea el entorno virtual en .venv e instala automáticamente las dependencias necesarias.

---

## Ejecución

El programa se ejecuta desde consola utilizando uv run:

### Ejercicio 1
#### And
```bash
uv run .\exer1\and.py <config_file_path>
```
#### Xor
```bash
uv run .\exer1\xor.py <config_file_path>
```

##### Archivo de configuración

Los programas reciben un archivo de configuración en formato JSON, en el cual se pueden indicar los siguientes parametros:

- learning_rate
- epochs

### Ejercicio 2
#### Entrenamiento

```bash
uv run .\exer2\learn.py config_file_path
```

#### Generalización

```bash
uv run .\exer2\generalize.py config_file_path
```

#### Archivo de configuración

El programa recibe un archivo de configuración en formato JSON, en el cual se indican los siguientes parametros:
- learning_rate
- epochs
- perceptron (lineal, tanh o sigmoid)
- partition_count (para la generalización solamente)

### Ejercicio 3
#### Xor
```bash
uv run .\exer3\mlp_xor_test.py --config <config_file_path>
```
#### Discriminacion de paridad
```bash
uv run .\exer3\mlp_pair_numbers.py --config <config_file_path>
```
##### Archivo de configuración

El programa recibe un archivo de configuración en formato JSON, en el cual se pueden indicar los siguientes parametros:

- learning_rate
- max_epochs
- activator_function
- optimizer
#### Discriminacion de digitos
```bash
uv run .\exer3\mlp_classify_numbers_images.py --config <config_file_path>
```
##### Archivo de configuración

El programa recibe un archivo de configuración en formato JSON, en el cual se pueden indicar los siguientes parametros:

- learning_rate
- max_epochs
- activator_function
- optimizer

### Las posibles funciones de activacion son:
- sigmoid
- tanh
- relu
- softplus
- mish
- leaky_relu

### Los posibles optimizadores son:
- gradient
- adam
- momentum