import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from utils.paths import generar_folder_fecha

@dataclass
class RumaInfo:
    id: int
    percent: float
    centroid: Tuple[int, int]
    radius: float


@dataclass
class AlertContext:
    frame: np.ndarray
    frame_count: int
    fps: float
    camera_sn: str
    enterprise: str = "default"
    ruma_summary: Optional[dict] = None
    frame_shape: Optional[Tuple[int, int]] = None
    detection_zone: Optional[List[Tuple[int, int]]] = None


def save_alert_local(
    alert_type: str,
    ruma_data: RumaInfo = None,
    context: AlertContext = None
):
    """Guarda localmente una alerta con imagen y metadatos"""
    timestamp = datetime.now()
    base_path = generar_folder_fecha("alerts_save", etiqueta="local")
    print("💾 Save alert local ejecutándose")

    # Calcular tiempo del video
    video_time_seconds = context.frame_count / context.fps

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
    if context.ruma_summary and context.frame_shape is not None:
        save_ruma_summary_image(
            ruma_summary=context.ruma_summary,
            frame_shape=context.frame_shape,
            base_path=base_path,
            timestamp=timestamp,
            frame_count=context.frame_count,
            detection_zone=context.detection_zone
        )

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
