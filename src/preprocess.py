"""
Módulo de Preprocesamiento Integral.
Etapas:
1. Procesamiento de Imagen (HSV Masking + CLAHE + ROI).
2. Sanitización (Conversión a JPG Fondo Negro + Renombrado Secuencial).
3. Inyección de Metadatos (Restauración de EXIF originales).
"""

import cv2
import numpy as np
import os
import shutil
import subprocess
from glob import glob
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_DIR = os.path.abspath(".")
INPUT_DIR = os.path.join(PROJECT_DIR, "data/raw")
PROCESSED_TEMP_DIR = os.path.join(PROJECT_DIR, "data/processed_temp") # Carpeta intermedia
OUTPUT_FINAL_DIR = os.path.join(PROJECT_DIR, "data/sanitized") # Salida final para AliceVision

# --- PARÁMETROS DE VISIÓN ---
# Ajustar según src/tuner.py
BG_LOWER = np.array([69, 173, 62])
BG_UPPER = np.array([92, 255, 255])
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
ROI_RATIO = 0.85 

def apply_clahe(image_bgr):
    """Aplica CLAHE al canal de luminosidad (L)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def create_circular_mask(h, w, ratio):
    """Genera máscara ROI circular."""
    center = (w // 2, h // 2)
    radius = int(min(h, w) * ratio / 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def keep_largest_component(mask):
    """Conserva solo el componente conexo más grande (el objeto)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2: return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_label] = 255
    return cleaned_mask

def stage_1_image_processing():
    """
    Etapa 1: Genera imágenes PNG con transparencia (RGBA) en una carpeta temporal.
    Retorna un diccionario mapeando {nombre_original: ruta_temporal_png}.
    """
    print("\n[Etapa 1/3] Procesamiento de Imagen (Masking + CLAHE)...")
    
    if os.path.exists(PROCESSED_TEMP_DIR):
        shutil.rmtree(PROCESSED_TEMP_DIR)
    os.makedirs(PROCESSED_TEMP_DIR)
    
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(INPUT_DIR, ext)))
    
    if not files:
        print(f"[ERROR] No hay imágenes en {INPUT_DIR}")
        return {}

    temp_map = {} # Clave: Nombre base original, Valor: Ruta PNG generada

    for file_path in tqdm(files, unit="img", desc="Procesando"):
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        
        img = cv2.imread(file_path)
        if img is None: continue
        
        h, w = img.shape[:2]

        # Lógica de Visión
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_fg = cv2.bitwise_not(cv2.inRange(hsv, BG_LOWER, BG_UPPER))
        mask_roi = create_circular_mask(h, w, ROI_RATIO)
        final_mask = cv2.bitwise_and(mask_fg, mask_roi)
        final_mask = keep_largest_component(final_mask)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        img_enhanced = apply_clahe(img)
        b, g, r = cv2.split(img_enhanced)
        final_img = cv2.merge([b, g, r, final_mask]) 

        output_path = os.path.join(PROCESSED_TEMP_DIR, f"{name}.png")
        cv2.imwrite(output_path, final_img)
        
        temp_map[name] = output_path
        
    return temp_map

def stage_2_sanitization(temp_map):
    """
    Etapa 2: Convierte PNGs temporales a JPGs (fondo negro) y renombra a secuencia (1000.jpg...).
    Retorna mapeo {nombre_original_base: ruta_final_jpg}.
    """
    print("\n[Etapa 2/3] Sanitización (Conversión a JPG + Renombrado)...")
    
    if os.path.exists(OUTPUT_FINAL_DIR):
        shutil.rmtree(OUTPUT_FINAL_DIR)
    os.makedirs(OUTPUT_FINAL_DIR)

    final_map = {}
    count = 0
    start_index = 1000

    # Ordenamos por nombre original para mantener la secuencia del video/fotos
    sorted_original_names = sorted(temp_map.keys())

    for original_name in tqdm(sorted_original_names, unit="file", desc="Sanitizando"):
        png_path = temp_map[original_name]
        
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        # Aplanar transparencia sobre fondo negro
        if len(img.shape) > 2 and img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
            bg = [cv2.bitwise_and(c, c, mask=a) for c in [b, g, r]]
            img_jpg = cv2.merge(bg)
        else:
            img_jpg = img

        new_filename = f"{start_index + count}.jpg"
        final_path = os.path.join(OUTPUT_FINAL_DIR, new_filename)
        
        cv2.imwrite(final_path, img_jpg)
        
        final_map[original_name] = final_path
        count += 1
        
    return final_map

def stage_3_metadata_injection(final_map):
    """
    Etapa 3: Copia metadatos EXIF del RAW original al JPG sanitizado usando exiftool.
    """
    print("\n[Etapa 3/3] Inyección de Metadatos EXIF...")
    
    # Mapear rutas completas de archivos RAW
    raw_files = glob(os.path.join(INPUT_DIR, '*'))
    # Diccionario: {nombre_base: ruta_completa_raw}
    raw_path_map = {os.path.splitext(os.path.basename(f))[0]: f for f in raw_files}

    processed_count = 0
    
    # Verificar si existe exiftool
    if shutil.which("exiftool") is None:
        print("[ERROR] 'exiftool' no está instalado o en el PATH. Se omitirá la inyección de metadatos.")
        print("        Esto puede causar que AliceVision falle al no detectar el sensor.")
        return

    for original_name, final_path in tqdm(final_map.items(), unit="img", desc="Inyectando"):
        raw_path = raw_path_map.get(original_name)
        
        if raw_path:
            cmd = [
                "exiftool", 
                "-overwrite_original", 
                "-TagsFromFile", raw_path, 
                "-all:all", 
                final_path
            ]
            # Ejecución silenciosa
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                processed_count += 1
    
    print(f"[INFO] Metadatos restaurados en {processed_count} imágenes.")

def main():
    print("="*60)
    print(" PREPARACIÓN DE DATOS PARA FOTOGRAMETRÍA")
    print("="*60)
    
    # 1. Procesamiento Visual
    temp_map = stage_1_image_processing()
    if not temp_map:
        return

    # 2. Normalización de Archivos
    final_map = stage_2_sanitization(temp_map)

    # 3. Restauración de Datos del Sensor
    stage_3_metadata_injection(final_map)

    # Limpieza (Opcional: borrar carpeta temporal)
    # shutil.rmtree(PROCESSED_TEMP_DIR) 

    print("\n" + "="*60)
    print(" [LISTO] Las imágenes están preparadas en: data/sanitized")
    print("         Ahora puede ejecutar 'src/pipeline.py'")
    print("="*60)

if __name__ == "__main__":
    main()