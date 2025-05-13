import sys

def calcular_accuracy(archivo):
    try:
        with open(archivo, 'r') as f:
            lineas = f.readlines()

            if len(lineas) <= 1:
                print("⚠️ El archivo está vacío o solo tiene el encabezado.")
                return

            total = 0
            aciertos = 0

            for linea in lineas[1:]:  # Ignorar la primera línea
                partes = linea.strip().split(',')
                if len(partes) != 2:
                    continue
                esperado, predicho = partes
                if esperado == predicho:
                    aciertos += 1
                total += 1

            if total == 0:
                print("⚠️ No hay datos válidos para calcular accuracy.")
                return

            accuracy = (aciertos / total) * 100
            print(f"✅ Accuracy: {accuracy:.2f}% ({aciertos}/{total})")
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python calcular_accuracy.py <archivo>")
    else:
        calcular_accuracy(sys.argv[1])
