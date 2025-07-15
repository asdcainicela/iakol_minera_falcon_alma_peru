import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Lista de imágenes ===
imagenes = [
    ("Mapa1_nuevo.png", "Mapa 1"),
    ("Mapa2_nuevo.png", "Mapa 2"),
    ("Mapa3_nuevo.png", "Mapa 3")
]

# === Ruta base ===
ruta_base = "homography/img/"

# === Procesar cada imagen ===
for nombre_archivo, titulo in imagenes:
    path = ruta_base + nombre_archivo
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] No se pudo cargar {path}")
        continue

    # Escalar a 1920x1080
    #img = cv2.resize(img, (1920, 1080))

    # Detección de bordes y contornos
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[ADVERTENCIA] No se encontraron contornos en {nombre_archivo}")
        continue

    largest_contour = max(contours, key=lambda c: cv2.arcLength(c, True))
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Dibujar los vértices
    img_poly = img.copy()
    for i, pt in enumerate(approx):
        coord = tuple(pt[0])
        cv2.circle(img_poly, coord, 5, (0, 255, 0), -1)
        cv2.putText(img_poly, f"{i+1}", coord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Mostrar resultado (una imagen a la vez)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_poly, cv2.COLOR_BGR2RGB))
    plt.title(f"{titulo} - {len(approx)} vértices")
    plt.axis("off")
    plt.show()

    # Imprimir coordenadas
    vertices = [pt[0].tolist() for pt in approx]
    print(f"=== {titulo} ===")
    print(f"Coordenadas de los vértices ({len(vertices)} puntos):")
    print(vertices)
    print()
