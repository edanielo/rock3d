"""
Script de configuración del entorno de desarrollo para Photogrammetry Pipeline.

Este script valida la instalación de AliceVision, actualiza la base de datos
de sensores de cámara locales y genera el script de variables de entorno necesario
para la ejecución del pipeline.

Uso:
    python setup_env.py
"""

import os
import sys
import logging
from pathlib import Path

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CONSTANTES ---
CURRENT_DIR = Path(os.getcwd())
# Ruta relativa estándar al framework AliceVision
ALICEVISION_ROOT = CURRENT_DIR / "aliceVision"
VARS_FILENAME = "vars.sh"

# Configuración del Sensor (Ej. Samsung S22 Ultra)
# TODO: Mover a un archivo de configuración externo (YAML/JSON) en una futura iteración.
CAMERA_MODEL_ID = "Samsung S22 Ultra"
SENSOR_WIDTH_MM = "9.6"  # Sensor 1/1.33"

def validate_binaries(root_path: Path) -> Path:
    """Valida la existencia de los binarios de AliceVision."""
    bin_path = root_path / "aliceVision" / "bin"
    if not bin_path.exists():
        bin_path = root_path / "bin"
    
    if not bin_path.exists():
        logger.error(f"No se encontró la carpeta 'bin' en {root_path}.")
        logger.error("Verifique la instalación del framework.")
        sys.exit(1)
    
    logger.info(f"Binarios validados en: {bin_path}")
    return bin_path

def update_sensor_database(root_path: Path):
    """
    Localiza y actualiza la base de datos de sensores de cámara si el modelo actual no existe.
    
    Args:
        root_path (Path): Ruta raíz de la instalación de AliceVision.
    """
    share_path = root_path / "aliceVision" / "share" / "aliceVision"
    db_file = share_path / "cameraSensors.db"
    
    if not db_file.exists():
        # Fallback para estructuras de directorios alternativas
        db_file = root_path / "share" / "aliceVision" / "cameraSensors.db"

    if not db_file.exists():
        logger.critical("No se encontró 'cameraSensors.db'. Imposible continuar.")
        sys.exit(1)

    logger.info(f"Base de datos de sensores: {db_file}")

    try:
        content = db_file.read_text(encoding='utf-8')
        
        if CAMERA_MODEL_ID in content:
            logger.info(f"Modelo '{CAMERA_MODEL_ID}' ya registrado.")
        else:
            logger.info(f"Registrando nuevo sensor: {CAMERA_MODEL_ID} ({SENSOR_WIDTH_MM}mm)")
            with open(db_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{CAMERA_MODEL_ID};{SENSOR_WIDTH_MM}\n")
                # Alias comunes para robustez
                f.write(f"SM-S908B;{SENSOR_WIDTH_MM}\n") 
                f.write(f"Samsung;SM-S908B;{SENSOR_WIDTH_MM}\n")
            logger.info("Base de datos actualizada correctamente.")
            
    except PermissionError:
        logger.warning("Permiso denegado en cameraSensors.db. Ejecute con 'sudo' si es necesario.")
    
    return db_file

def generate_env_script(bin_path: Path, db_file: Path, root_path: Path):
    """Genera el script shell para exportar variables de entorno."""
    env_script = CURRENT_DIR / VARS_FILENAME
    lib_path = root_path / "aliceVision" / "lib"
    av_root_internal = root_path / "aliceVision"

    content = (
        "# Script de entorno autogenerado\n"
        f"export ALICEVISION_BIN='{bin_path}'\n"
        f"export ALICEVISION_SENSOR_DB='{db_file}'\n"
        f"export ALICEVISION_ROOT='{av_root_internal}'\n"
        f"export LD_LIBRARY_PATH='{lib_path}:$LD_LIBRARY_PATH'\n"
    )

    try:
        env_script.write_text(content, encoding='utf-8')
        logger.info(f"Archivo de entorno generado: {env_script.name}")
        print(f"\n[ACCIÓN REQUERIDA] Ejecute: source {env_script.name}\n")
    except IOError as e:
        logger.error(f"Error escribiendo {env_script.name}: {e}")

def main():
    logger.info("Iniciando configuración de infraestructura...")
    
    if not ALICEVISION_ROOT.exists():
        logger.error(f"Directorio base no encontrado: {ALICEVISION_ROOT}")
        sys.exit(1)

    bin_path = validate_binaries(ALICEVISION_ROOT)
    db_file = update_sensor_database(ALICEVISION_ROOT)
    generate_env_script(bin_path, db_file, ALICEVISION_ROOT)

if __name__ == "__main__":
    main()