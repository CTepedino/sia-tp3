import numpy as np
import pandas as pd

# Cargar el dataset
df = pd.read_csv("TP3-ej2-conjunto.csv")

# Separar características y target
X = df[['x1', 'x2', 'x3']].values
y = df['y'].values

# Escalar X y y dividiendo por el valor absoluto máximo de cada uno
max_valor_X = np.abs(X).max(axis=0)  # Valor máximo por columna de X
max_valor_y = np.abs(y).max()  # Valor máximo de y

X_escalado = X / max_valor_X  # Escalar cada columna de X por su propio valor máximo
y_escalado = y / max_valor_y  # Escalar y por su valor máximo

# Crear nuevo DataFrame con datos escalados
df_escalado = pd.DataFrame(X_escalado, columns=['x1', 'x2', 'x3'])
df_escalado['y'] = y_escalado  # Escalar y

# Guardar a nuevo CSV
df_escalado.to_csv("TP3-ej2-escalado.csv", index=False)

print(f"Dataset escalado guardado como 'TP3-ej2-escalado.csv'. Valor máximo de X: {max_valor_X}, Valor máximo de y: {max_valor_y}")
