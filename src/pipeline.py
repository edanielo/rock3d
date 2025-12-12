"""
Orquestador del Pipeline de Fotogrametría (AliceVision).
Asume que los datos ya han sido preparados por src/preprocess.py.

Autor: [Tu Nombre]
"""

import os
import subprocess
import sys
import shutil

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_DIR = os.path.abspath(".")
# Input directo: la carpeta sanitizada
INPUT_IMAGES_DIR = os.path.join(PROJECT_DIR, "data/sanitized") 
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "data/recon")

# Verificación de Variables de Entorno
ALICEVISION_BIN = os.environ.get("ALICEVISION_BIN")
SENSOR_DB = os.environ.get("ALICEVISION_SENSOR_DB")

def validate_environment():
    """Verifica pre-condiciones críticas antes de iniciar."""
    if not ALICEVISION_BIN or not SENSOR_DB:
        print("[ERROR] Variables de entorno no configuradas.")
        print("        Ejecute 'source vars.sh' antes de iniciar.")
        return False

    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"[ERROR] No se encuentra el directorio de entrada: {INPUT_IMAGES_DIR}")
        print("        Por favor ejecute primero 'python src/preprocess.py'.")
        return False

    num_files = len([f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith('.jpg')])
    if num_files < 3:
        print(f"[ERROR] Insuficientes imágenes en {INPUT_IMAGES_DIR} (Encontradas: {num_files}).")
        print("        Se requieren al menos 3 imágenes para la triangulación.")
        return False

    return True

def run_alicevision_node(cmd_list, step_name):
    """Ejecuta un subproceso de AliceVision gestionando la salida estándar."""
    print(f"\n[Ejecución] Nodo: {step_name}")
    try:
        # verboseLevel 'error' para mantener la consola limpia
        if "--verboseLevel" not in cmd_list:
             cmd_list.extend(["--verboseLevel", "error"])
             
        subprocess.run(cmd_list, check=True, text=True)
        print(f"[Estado] {step_name}: Completado.")
    except subprocess.CalledProcessError:
        print(f"\n[ERROR FATAL] Fallo en el nodo: {step_name}")
        print(f"Comando: {' '.join(cmd_list)}")
        sys.exit(1)

def main():
    print("="*60)
    print(" SISTEMA DE RECONSTRUCCIÓN 3D - ALICEVISION PIPELINE")
    print("="*60)

    if not validate_environment():
        sys.exit(1)
    
    # Limpieza de reconstrucciones previas
    if os.path.exists(OUTPUT_ROOT):
        print("[INFO] Limpiando directorio de salida previo...")
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # --- EJECUCIÓN DEL FLUJO DE TRABAJO ---
    
    # 1. Inicialización de Cámara
    sfm_file = os.path.join(OUTPUT_ROOT, "cameraInit.sfm")
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_cameraInit"),
        "--imageFolder", INPUT_IMAGES_DIR,
        "--sensorDatabase", SENSOR_DB,
        "--output", sfm_file,
        "--allowSingleView", "1",
        "--viewIdMethod", "filename" # Importante: usa los nombres 1000, 1001...
    ], "CameraInit")

    # 2. Extracción de Características (SIFT)
    features_dir = os.path.join(OUTPUT_ROOT, "features")
    os.makedirs(features_dir, exist_ok=True)
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_featureExtraction"),
        "--input", sfm_file,
        "--output", features_dir,
        "--describerTypes", "sift",
        "--forceCpuExtraction", "1"
    ], "FeatureExtraction")

    # 3. Correspondencia de Imágenes
    matches_file = os.path.join(OUTPUT_ROOT, "imageMatches.txt")
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_imageMatching"),
        "--input", sfm_file,
        "--features", features_dir,
        "--output", matches_file,
        "--minNbImages", "2",
        "--method", "Sequential", 
        "--nbNeighbors", "50"
    ], "ImageMatching")

    # 4. Correspondencia de Características
    matches_dir = os.path.join(OUTPUT_ROOT, "matches")
    os.makedirs(matches_dir, exist_ok=True)
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_featureMatching"),
        "--input", sfm_file,
        "--features", features_dir,
        "--imagePairsList", matches_file,
        "--output", matches_dir,
        "--describerTypes", "sift",
        "--distanceRatio", "0.8"
    ], "FeatureMatching")

    # 5. Estructura a partir de Movimiento (SfM)
    sfm_output = os.path.join(OUTPUT_ROOT, "sfm.abc")
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_incrementalSfM"),
        "--input", sfm_file,
        "--features", features_dir,
        "--matches", matches_dir,
        "--output", sfm_output,
        "--outputViewsAndPoses", os.path.join(OUTPUT_ROOT, "cameras.sfm"),
    ], "StructureFromMotion")

    # 6. Preparación de Escena Densa
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_prepareDenseScene"),
        "--input", sfm_output,
        "--output", OUTPUT_ROOT,
        "--imagesFolders", INPUT_IMAGES_DIR
    ], "PrepareDenseScene")

    # 7. Estimación de Mapa de Profundidad
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_depthMapEstimation"),
        "--input", sfm_output,
        "--output", OUTPUT_ROOT,
        "--imagesFolder", INPUT_IMAGES_DIR,
        "--downscale", "2"
    ], "DepthMapEstimation")

    # 8. Filtrado de Mapas de Profundidad
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_depthMapFiltering"),
        "--input", sfm_output,
        "--output", OUTPUT_ROOT,
        "--depthMapsFolder", OUTPUT_ROOT
    ], "DepthMapFiltering")

    # 9. Generación de Malla (Meshing)
    dense_sfm = os.path.join(OUTPUT_ROOT, "densePointCloud.sfm")
    mesh_output = os.path.join(OUTPUT_ROOT, "mesh.obj")
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_meshing"),
        "--input", sfm_output,
        "--output", dense_sfm,
        "--outputMesh", mesh_output,
        "--depthMapsFolder", OUTPUT_ROOT,
        "--minVis", "2"
    ], "Meshing")

    # 10. Texturizado
    run_alicevision_node([
        os.path.join(ALICEVISION_BIN, "aliceVision_texturing"),
        "--input", dense_sfm,
        "--inputMesh", mesh_output,
        "--output", OUTPUT_ROOT,
        "--imagesFolder", INPUT_IMAGES_DIR,
        "--textureSide", "4096",
        "--colorMappingFileType", "png"
    ], "Texturing")

    print("\n" + "="*60)
    print(" [INFO] PROCESO FINALIZADO EXITOSAMENTE")
    print(f" [INFO] Modelo generado en: {os.path.join(OUTPUT_ROOT, 'texturedMesh.obj')}")
    print("="*60)

if __name__ == "__main__":
    main()