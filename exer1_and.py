import os
import sys
import cv2

def crear_video(input_path, output_filename="video_output.mp4", fps=5):
    archivos = sorted(
        [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    if not archivos:
        print("No se encontraron imágenes en el directorio.")
        return

    primer_frame = cv2.imread(os.path.join(input_path, archivos[0]))
    if primer_frame is None:
        print("No se pudo leer la primera imagen.")
        return

    height, width, _ = primer_frame.shape

    # Usa AVI seguro
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if not output_filename.endswith('.avi'):
        output_filename = output_filename.rsplit('.', 1)[0] + '.avi'

    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for archivo in archivos:
        img_path = os.path.join(input_path, archivo)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Advertencia: no se pudo leer {img_path}. Saltando.")
            continue

        # Redimensionar por si hay algún frame de distinto tamaño
        frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()
    print(f"Video guardado como: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python crear_video.py <carpeta_imagenes> [nombre_salida]")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_filename = sys.argv[2]
    else:
        output_filename = "video_output.avi"

    crear_video(input_path, output_filename)
