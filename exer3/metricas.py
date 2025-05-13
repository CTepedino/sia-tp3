import sys
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import pandas as pd

def cargar_datos(archivo):
    with open(archivo, 'r') as f:
        lineas = f.readlines()[1:]  # Omitir encabezado
        reales = []
        predichos = []
        for linea in lineas:
            partes = linea.strip().split(',')
            if len(partes) == 2:
                pred, real = partes
                try:
                    reales.append(int(real))
                    predichos.append(int(pred))
                except ValueError:
                    continue
        return reales, predichos

def calcular_metricas(reales, predichos):
    etiquetas = list(range(10))  # 0 al 9
    matriz = confusion_matrix(reales, predichos, labels=etiquetas)
    acc = accuracy_score(reales, predichos)
    precision = precision_score(reales, predichos, average=None, labels=etiquetas, zero_division=0)
    recall = recall_score(reales, predichos, average=None, labels=etiquetas, zero_division=0)
    f1 = f1_score(reales, predichos, average=None, labels=etiquetas, zero_division=0)

    print("\n✅ Matriz de Confusión:")
    df = pd.DataFrame(matriz, index=[f"Real {i}" for i in etiquetas], columns=[f"Pred {i}" for i in etiquetas])
    print(df)

    print(f"\n🔢 Accuracy: {acc*100:.2f}%")

    print("\n📊 Métricas por clase:")
    for i in etiquetas:
        print(f"Clase {i}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}, F1-Score = {f1[i]:.2f}")

    print("\n📝 Reporte completo:")
    print(classification_report(reales, predichos, labels=etiquetas, zero_division=0))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python metricas_multiclase.py <archivo>")
    else:
        reales, predichos = cargar_datos(sys.argv[1])
        calcular_metricas(reales, predichos)
