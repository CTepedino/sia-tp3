import pandas as pd

# Ruta al archivo
file_path = './exer3/results/resultados_ej3b.txt'

# Leer el archivo
df = pd.read_csv(file_path)

# Crear tabla de contingencia
tabla = pd.crosstab(df['prediccion'], df['resultado'])

# Mostrar tabla
print("Tabla de ocurrencias (filas=obtenido, columnas=esperado):\n")
print(tabla)
