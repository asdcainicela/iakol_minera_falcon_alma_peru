import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

from alerts.alert_info import RumaInfo, AlertContext
from utils.paths import generar_folder_fecha

def save_alert_local(
    alert_type: str,
    ruma_data: RumaInfo = None,
    context: AlertContext = None
):
    """Guarda localmente una alerta con imagen y metadatos"""
    timestamp = datetime.now()
    base_path = generar_folder_fecha("alerts_save", etiqueta="local")
    #print(" Save alert local ejecutándose")

    # Calcular tiempo del video
    video_time_seconds = context.frame_count / context.fps

    if ruma_data and ruma_data.centroid_homographic is not None and ruma_data.radius_homographic is not None:
        centroid = ruma_data.centroid_homographic
        radius = ruma_data.radius_homographic

        if ruma_data.percent == 100 and alert_type == 'nueva_ruma':
            # Agrega datos transformados al resumen si aún no están
            context.ruma_summary[ruma_data.id]['centroid_homographic'] = centroid
            context.ruma_summary[ruma_data.id]['radius_homographic'] = radius

            save_ruma_summary_image_homography(
                ruma_summary=context.ruma_summary,
                frame_shape=context.frame_shape,
                base_path=base_path,
                timestamp=timestamp,
                frame_count=context.frame_count,
                detection_zone=context.detection_zone,
                map_image_path="homography/img/Mapa1_nuevo.png"
            )

            save_ruma_summary_image(
                ruma_summary=context.ruma_summary,
                frame_shape=context.frame_shape,
                base_path=base_path,
                timestamp=timestamp,
                frame_count=context.frame_count,
                detection_zone=context.detection_zone
            )

    # Metadata de la alerta
    metadata = {
        "cameraSN": context.camera_sn,
        "enterprise": context.enterprise,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "id": ruma_data.id, 
        "percent": ruma_data.percent,
        "coords": ruma_data.centroid,
        "radius": ruma_data.radius,
        "centroid_homographic": ruma_data.centroid_homographic,
        "radius_homographic": ruma_data.radius_homographic,
        "frame": None,
        "frame_number": context.frame_count,
        "video_time_seconds": video_time_seconds,
    }
    #print(f" Metadata de alerta: {metadata}")

    # Nombres de archivo
    base_filename = f"{timestamp.strftime('%H-%M-%S')}_{alert_type}_{context.frame_count}"
    json_path = base_path / f"{base_filename}.json"
    image_path = base_path / f"{base_filename}.jpg"

    # Guardar JSON y frame
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    cv2.imwrite(str(image_path), context.frame)

    # Guardar resumen visual si hay nueva ruma
    """
    if context.ruma_summary and context.frame_shape is not None:
        
    """
    #print(f" Alerta local guardada: {alert_type} - {timestamp.strftime('%H:%M:%S')}")


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
    cv2.imwrite(str(save_path), summary_image)
    #print(f" Imagen resumen de rumas guardada en: {save_path}")

def save_ruma_summary_image_homography(
    ruma_summary,
    frame_shape,
    base_path,
    timestamp,
    frame_count,
    detection_zone=None,
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
    cv2.imwrite(str(save_path), mapa)
    #print(f" Imagen homográfica de rumas guardada en: {save_path}")
