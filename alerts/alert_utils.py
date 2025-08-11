import cv2
import numpy as np

def save_ruma_summary_image(
    ruma_summary,
    frame_shape,
    base_path,
    timestamp,
    frame_count,
    detection_zone=None
    ):
    """Guarda imagen resumen de rumas (puntos y radios)"""
    summary_image = np.ones(frame_shape, dtype=np.uint8) * 255  # fondo blanco

    # Dibujar zona de detección si existe
    if detection_zone is not None:
        pts = np.array(detection_zone).reshape((-1, 1, 2))
        cv2.polylines(summary_image, [pts], isClosed=True, color=(200, 200, 200), thickness=2)

    for ruma_id, info in ruma_summary.items():
        centroid = info['centroid']
        radius = int(info['radius'])

        cv2.circle(summary_image, centroid, radius, (0, 0, 255), 2)  # círculo rojo
        cv2.circle(summary_image, centroid, 3, (0, 0, 0), -1)        # centroide negro

        label_pos = (centroid[0] + 10, centroid[1] - 10)
        cv2.putText(summary_image, f"R{ruma_id}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    filename = f"{timestamp.strftime('%H-%M-%S')}_ruma_summary_{frame_count}.jpg"
    save_path = base_path / filename
    if summary_image is not None :
        cv2.imwrite(str(save_path), summary_image)
    else:
        print(f"[Error] El frame está vacío. No se pudo guardar la imagen en: {save_path}")
    
    #print(f" Imagen resumen de rumas guardada en: {save_path}")

def save_ruma_summary_image_homography(
    ruma_summary,
    base_path,
    timestamp,
    frame_count,
    map_image_path: str = "homography/img/Mapa1_nuevo.png",
):
    """Dibuja todas las rumas transformadas por homografía sobre el mapa y guarda la imagen."""
    mapa = cv2.imread(map_image_path)
    if mapa is None:
        print(f"[ERROR] No se pudo cargar la imagen: {map_image_path}")
        return

    # Dibujar todas las rumas transformadas
    for ruma_id, info in ruma_summary.items():
        centroid_h = info.get('centroid_homographic')
        radius_h = info.get('radius_homographic')

        if centroid_h is None or radius_h is None:
            continue  # Saltar si falta información

        # Asegurar valores válidos
        cx, cy = int(round(centroid_h[0])), int(round(centroid_h[1]))
        radius = int(round(radius_h))

        cv2.circle(mapa, (cx, cy), radius, (0, 0, 255), 2)  # círculo rojo
        cv2.circle(mapa, (cx, cy), 3, (0, 0, 0), -1)        # centroide negro

        label_pos = (cx + 10, cy - 10)
        cv2.putText(mapa, f"R{ruma_id}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Guardar imagen
    filename = f"{timestamp.strftime('%H-%M-%S')}_ruma_summary_homography_{frame_count}.jpg"
    save_path = base_path / filename
    #cv2.imwrite(str(save_path), mapa)
    if mapa is not None :
        cv2.imwrite(str(save_path), mapa)
    else:
        print(f"[Error] El frame está vacío. No se pudo guardar la imagen en: {save_path}")
    #print(f" Imagen homográfica de rumas guardada en: {save_path}")
