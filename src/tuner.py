"""
Herramienta de calibración de umbrales HSV.
Permite la visualización en tiempo real de la máscara de segmentación.
"""

import cv2
import numpy as np
import os
from glob import glob

# --- CONFIGURACIÓN ---
INPUT_DIR = "data/raw"
WINDOW_NAME = "Calibrador HSV"

def nothing(x):
    pass

def tuner():
    """Ejecuta la interfaz gráfica para ajuste de parámetros HSV."""
    
    files = sorted(glob(os.path.join(INPUT_DIR, '*')))
    valid_exts = ('.jpg', '.jpeg', '.png')
    files = [f for f in files if f.lower().endswith(valid_exts)]
    
    if not files:
        print(f"[ERROR] No se encontraron imágenes válidas en el directorio {INPUT_DIR}")
        return
    
    # Selección de muestra representativa (imagen central de la secuencia)
    sample_img_path = files[len(files)//2]
    print(f"[INFO] Cargando imagen de muestra: {sample_img_path}")
    
    img = cv2.imread("/home/edani/Documents/UADY/007-semestre/Video/final_project/rock3d/data/raw/20251127_133354.jpg")
    if img is None:
        print(f"[ERROR] Fallo al leer el archivo de imagen.")
        return

    # Ajuste de resolución para visualización
    target_height = 600
    h, w = img.shape[:2]
    scale = target_height / h
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.namedWindow(WINDOW_NAME)

    # Inicialización de controles deslizantes (Trackbars)
    cv2.createTrackbar("H - Min", WINDOW_NAME, 35, 179, nothing)
    cv2.createTrackbar("H - Max", WINDOW_NAME, 85, 179, nothing)
    cv2.createTrackbar("S - Min", WINDOW_NAME, 50, 255, nothing)
    cv2.createTrackbar("S - Max", WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar("V - Min", WINDOW_NAME, 50, 255, nothing)
    cv2.createTrackbar("V - Max", WINDOW_NAME, 255, 255, nothing)

    print("[INFO] Interfaz iniciada. Ajuste los valores para aislar el objeto.")
    print("[INFO] Presione 'q' para guardar los valores y salir.")

    while True:
        l_h = cv2.getTrackbarPos("H - Min", WINDOW_NAME)
        u_h = cv2.getTrackbarPos("H - Max", WINDOW_NAME)
        l_s = cv2.getTrackbarPos("S - Min", WINDOW_NAME)
        u_s = cv2.getTrackbarPos("S - Max", WINDOW_NAME)
        l_v = cv2.getTrackbarPos("V - Min", WINDOW_NAME)
        u_v = cv2.getTrackbarPos("V - Max", WINDOW_NAME)

        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        
        # Visualización: Fondo eliminado
        background_removed = cv2.bitwise_and(img, img, mask=mask_inv)
        preview = np.hstack((img, background_removed))
        
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    # Salida de parámetros formateada para copiar/pegar
    print("\n" + "-"*50)
    print("[RESULTADOS] Parámetros de calibración:")
    print(f"BG_LOWER = np.array([{l_h}, {l_s}, {l_v}])")
    print(f"BG_UPPER = np.array([{u_h}, {u_s}, {u_v}])")
    print("-"*50 + "\n")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tuner()