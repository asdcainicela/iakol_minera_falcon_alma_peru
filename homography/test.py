import cv2
import numpy as np
from pathlib import Path

# === Datos de rumas transformadas (centros y radios)
centros = [
    (726.9493408203125, 382.6887512207031),
    (441.7439270019531, 298.60394287109375),
    (495.19256591796875, 548.8800659179688),
    (602.9282836914062, 381.5032653808594),
    (705.6707153320312, 316.296630859375),
    (560.7971801757812, 291.5372009277344)
]

radios = [
    16.04414939880371,
    47.49702453613281,
    29.544191360473633,
    42.850929260253906,
    49.6846809387207,
    49.767799377441406
]

# === Ruta de la imagen base
mapa_path = Path("homography/img/Mapa1_nuevo.png")
mapa = cv2.imread(str(mapa_path))

if mapa is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {mapa_path}")

# === Dibujar todas las rumas
for i, (centro, radio) in enumerate(zip(centros, radios), 1):
    centro_int = tuple(map(int, centro))
    radio_int = int(round(radio))

    cv2.circle(mapa, centro_int, radio_int, (0, 0, 255), 2)  # círculo rojo
    cv2.circle(mapa, centro_int, 3, (0, 0, 0), -1)           # centroide negro
    label_pos = (centro_int[0] + 10, centro_int[1] - 10)
    cv2.putText(mapa, f"R{i}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# === Guardar resultado
output_path = mapa_path.parent / "mapa_rumas_all.jpg"
cv2.imwrite(str(output_path), mapa)
print(f"✅ Imagen con todas las rumas guardada en: {output_path}")
