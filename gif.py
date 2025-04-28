import os
import re
import imageio.v2 as imageio

# Carpeta donde están las imágenes
input_folder = 'results'
output_gif = 'results/animation.gif'

# Función para extraer números de epoch e iteración
def extract_numbers(filename):
    match = re.search(r'epoch(\d+)_iter(\d+)', filename)
    if match:
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (epoch, iteration)
    else:
        return (float('inf'), float('inf'))  # Si no matchea, mandarlo al final

# Buscar y ordenar archivos .png
filenames = [f for f in os.listdir(input_folder) if f.endswith('.png')]
filenames.sort(key=extract_numbers)

# Leer imágenes
images = []
for filename in filenames:
    file_path = os.path.join(input_folder, filename)
    images.append(imageio.imread(file_path))

# Guardar GIF
imageio.mimsave(output_gif, images, duration=0.2)

print(f'GIF guardado en {output_gif}')
