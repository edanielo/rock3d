"""
Módulo de Preprocesamiento de Imágenes.

Implementa un pipeline de segmentación semántica utilizando U-2-Net (vía rembg)
para la eliminación de fondo, seguido de un algoritmo de supresión de color (despill)
y normalización de metadatos EXIF.
"""

import cv2
import numpy as np
import os
import shutil
import subprocess
import logging
from glob import glob
from pathlib import Path
from tqdm import tqdm
from rembg import remove, new_session

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
PROJECT_DIR = Path(os.getcwd())
INPUT_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "sanitized"

# Parámetros de Procesamiento
APPLY_DESPILL = True 
DESPILL_FACTOR = 0.8 

def green_spill_reduction(img_array: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
    """
    Reduce la contaminación de color (spill) en los bordes de la segmentación.
    
    Args:
        img_array: Imagen RGB original.
        mask_array: Máscara alpha generada por la segmentación.
    
    Returns:
        np.ndarray: Imagen corregida.
    """
    img_float = img_array.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    # Promedio de canales rojo y azul como referencia
    rb_avg = (r + b) / 2.0
    
    # Detección de zonas donde el verde excede el promedio RB dentro de la máscara
    spill_mask = (g > rb_avg) & (mask_array > 0)
    
    if np.any(spill_mask):
        g[spill_mask] = (g[spill_mask] * (1 - DESPILL_FACTOR)) + (rb_avg[spill_mask] * DESPILL_FACTOR)
        
    return cv2.merge([b, g, r]).astype(np.uint8)

def sanitize_images() -> dict:
    """
    Ejecuta el ciclo principal de limpieza: Inferencia IA -> Post-proceso -> Guardado.
    
    Returns:
        dict: Mapeo de {nombre_original: ruta_final}.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Búsqueda de imágenes (case insensitive)
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(list(INPUT_DIR.glob(ext)))
    
    if not files:
        logger.error(f"No se encontraron imágenes en {INPUT_DIR}")
        return {}

    files = sorted(files)
    logger.info(f"Cargando modelo de segmentación para {len(files)} imágenes...")

    try:
        session = new_session(model_name='u2net')
    except Exception as e:
        logger.critical(f"Error inicializando motor de IA: {e}")
        return {}

    mapping = {} 
    start_index = 1000 # Índice inicial estándar para secuencias de imágenes

    for i, file_path in enumerate(tqdm(files, desc="Segmentación IA")):
        filename = file_path.name
        original_name_base = file_path.stem
        
        try:
            with open(file_path, 'rb') as f:
                input_data = f.read()
                # Inferencia
                output_data = remove(input_data, session=session)
            
            # Decodificación segura
            img_rgba = cv2.imdecode(np.frombuffer(output_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            
            if img_rgba is None:
                logger.warning(f"Decodificación fallida para {filename}")
                continue

        except Exception as e:
            logger.error(f"Error procesando {filename}: {e}")
            continue

        # Separación de canales
        b, g, r, a = cv2.split(img_rgba)
        img_rgb = cv2.merge([b, g, r])
        
        # Post-procesamiento
        if APPLY_DESPILL:
            img_rgb = green_spill_reduction(img_rgb, a)

        # Composición sobre fondo negro (evita artefactos en fotogrametría)
        bg_black = np.zeros_like(img_rgb)
        alpha_f = a[:, :, np.newaxis] / 255.0
        comp_final = (img_rgb * alpha_f) + (bg_black * (1.0 - alpha_f))
        
        # Guardado
        new_filename = f"{start_index + i}.jpg"
        final_path = OUTPUT_DIR / new_filename
        
        cv2.imwrite(str(final_path), comp_final.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        mapping[original_name_base] = final_path

    return mapping

def transfer_metadata(mapping: dict):
    """
    Transfiere metadatos EXIF originales usando ExifTool y normaliza la orientación.
    
    Args:
        mapping (dict): Diccionario vinculando nombres originales con rutas procesadas.
    """
    logger.info("Iniciando transferencia y normalización de metadatos EXIF...")
    
    if shutil.which("exiftool") is None:
        logger.error("Herramienta 'exiftool' no encontrada en el sistema. Se omiten metadatos.")
        return

    # Mapa de búsqueda inversa para archivos raw
    raw_files = list(INPUT_DIR.glob('*'))
    raw_map = {f.stem: f for f in raw_files}
    
    success_count = 0
    
    for orig_name, san_path in tqdm(mapping.items(), desc="Inyección EXIF"):
        raw_path = raw_map.get(orig_name)
        if raw_path:
            cmd = [
                "exiftool", 
                "-overwrite_original", 
                "-TagsFromFile", str(raw_path), 
                "-all:all",           
                "-n",                 
                "-Orientation=1",     # Normalización crítica para fotogrametría
                str(san_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                success_count += 1
                
    logger.info(f"Metadatos procesados correctamente en {success_count}/{len(mapping)} archivos.")

def main():
    logger.info("Iniciando pipeline de preprocesamiento...")
    mapping = sanitize_images()
    if mapping:
        transfer_metadata(mapping)
        logger.info(f"Proceso finalizado. Salida disponible en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()