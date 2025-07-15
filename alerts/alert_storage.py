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
    context: AlertContext = None,
    transformer: Optional[object] = None
):
    """Guarda localmente una alerta con imagen y metadatos"""
    timestamp = datetime.now()
    base_path = generar_folder_fecha("alerts_save", etiqueta="local")
    print(" Save alert local ejecutándose")

    # Calcular tiempo del video
    video_time_seconds = context.frame_count / context.fps
    
    if ruma_data and ruma_data.centroid is not None and ruma_data.radius is not None:
        radius = float(ruma_data.radius)
        centroid = tuple(ruma_data.centroid)

        if transformer is not None:
            centroid, radius = transformer.transform_circle(centroid, radius)
            centroid = tuple(map(float, centroid))
            radius = float(radius)

            print(f"[INFO] Radio original: {ruma_data.radius:.2f}, Centroide original: {ruma_data.centroid}")
            print(f"[INFO] Radio transformado: {radius:.8f}, Centroide transformado: {centroid}")

            if ruma_data.percent == 100:
                draw_transformed_rumas_on_map(
                    centroid=centroid,
                    radius=radius,
                    ruma_id=ruma_data.id,
                    map_image_path="homography/img/Mapa1_nuevo.png",
                    output_folder=base_path,
                    frame_count=context.frame_count
                )

                save_ruma_summary_image(
                    ruma_summary=context.ruma_summary,
                    frame_shape=context.frame_shape,
                    base_path=base_path,
                    timestamp=timestamp,
                    frame_count=context.frame_count,
                    detection_zone=context.detection_zone
                )
        else:
            print(f"[INFO] Sin transformación: radio = {radius:.2f}, centroide = {centroid}")
    else:
        centroid = None
        radius = None

    # Metadata de la alerta
    metadata = {
        "cameraSN": context.camera_sn,
        "enterprise": context.enterprise,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "id": ruma_data.id, 
        "percent": ruma_data.percent,
        "coords": centroid, #ruma_data.centroid,
        "radius": radius,#ruma_data.radius,
        "frame": None,
        "frame_number": context.frame_count,
        "video_time_seconds": video_time_seconds,
    }
    print(f" Metadata de alerta: {metadata}")

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
    print(f" Alerta local guardada: {alert_type} - {timestamp.strftime('%H:%M:%S')}")


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
    print(f"📄 Imagen resumen de rumas guardada en: {save_path}")

def draw_transformed_rumas_on_map(
    centroid: Tuple[float, float],
    radius: float,
    ruma_id: int,
    map_image_path: str = "homography/img/Mapa1_nuevo.png",
    output_folder: Optional[Path] = None,
    frame_count: Optional[int] = None
):
    """Dibuja una ruma transformada sobre el mapa y guarda la imagen."""
    mapa = cv2.imread(map_image_path)
    if mapa is None:
        print(f"[ERROR] No se pudo cargar la imagen: {map_image_path}")
        return

    centro = tuple(map(int, centroid))
    radio = int(round(radius))

    # Dibujar círculo y centroide
    cv2.circle(mapa, centro, radio, (0, 0, 255), 2)
    cv2.circle(mapa, centro, 3, (0, 0, 0), -1)
    cv2.putText(mapa, f"R{ruma_id}", (centro[0] + 10, centro[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Guardar la imagen modificada
    now = datetime.now()
    folder = Path(map_image_path).parent if output_folder is None else output_folder
    frame_suffix = f"_{frame_count}" if frame_count else ""
    out_path = folder / f"mapa_con_ruma_{ruma_id}_{now.strftime('%H-%M-%S')}{frame_suffix}.jpg"
    cv2.imwrite(str(out_path), mapa)
    print(f"🗺️ Mapa con ruma R{ruma_id} guardado en: {out_path}")
