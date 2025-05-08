import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_number_image(number, size=(28, 28), font_size=20, noise_level=0.1, rotation_range=(-15, 15)):
    # Crear una imagen en blanco
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    
    # Intentar cargar una fuente, si no está disponible usar la default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calcular posición para centrar el número
    text_width = draw.textlength(str(number), font=font)
    text_height = font_size
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Dibujar el número
    draw.text((x, y), str(number), fill=0, font=font)
    
    # Aplicar rotación aleatoria
    angle = random.uniform(rotation_range[0], rotation_range[1])
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    
    # Aplicar ruido
    img_array = np.array(img)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img

def get_next_index(output_dir, digit):
    """Obtiene el siguiente índice disponible para un dígito"""
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(f"numero_{digit}_")]
    if not existing_files:
        return 0
    
    # Extraer los índices existentes
    indices = []
    for file in existing_files:
        try:
            idx = int(file.split('_')[2].split('.')[0])
            indices.append(idx)
        except:
            continue
    
    return max(indices) + 1 if indices else 0

def generate_training_set(output_dir="./utils/numeros", samples_per_number=50):
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    total_generated = 0
    
    # Generar imágenes para cada dígito
    for digit in range(10):
        print(f"Generando imágenes para el dígito {digit}...")
        start_index = get_next_index(output_dir, digit)
        remaining_samples = samples_per_number - start_index
        
        if remaining_samples <= 0:
            print(f"Ya existen suficientes imágenes para el dígito {digit}")
            continue
            
        for i in range(remaining_samples):
            # Variar parámetros para cada imagen
            noise = random.uniform(0.05, 0.2)
            rotation = random.uniform(-20, 20)
            font_size = random.randint(18, 22)
            
            # Crear imagen
            img = create_number_image(
                digit,
                noise_level=noise,
                rotation_range=(rotation, rotation),
                font_size=font_size
            )
            
            # Guardar imagen con el siguiente índice disponible
            current_index = start_index + i
            filename = f"numero_{digit}_{current_index:03d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            total_generated += 1
    
    print(f"\nSe generaron {total_generated} nuevas imágenes en {output_dir}")
    print(f"Total de imágenes en el directorio: {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    generate_training_set() 