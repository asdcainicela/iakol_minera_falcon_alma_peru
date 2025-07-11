import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def save_alert_local(
    alert_type,
    frame,
    frame_count,
    fps,
    camera_sn,
    enterprise="default",
    ruma_summary=None,
    frame_shape=None,
    detection_zone=None
):
    """Guarda localmente una alerta con imagen y metadatos"""
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    base_path = Path("alerts_save") / f"{date_str}_local"
    base_path.mkdir(parents=True, exist_ok=True) 
    print("💾 Save alert local ejecutándose")

    # Calcular tiempo del video
    video_time_seconds = frame_count / fps

    # Metadata de la alerta
    metadata = {
        "cameraSN": camera_sn,
        "enterprise": enterprise,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "video_time_seconds": video_time_seconds,
        "frame_number": frame_count
    }

    # Nombres de archivo
    base_filename = f"{timestamp.strftime('%H-%M-%S')}_{alert_type}_{frame_count}"
    json_path = base_path / f"{base_filename}.json"
    image_path = base_path / f"{base_filename}.jpg"

    # Guardar JSON y frame
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    cv2.imwrite(str(image_path), frame)

    # Guardar resumen visual si hay nueva ruma
    if ruma_summary and frame_shape is not None:
        save_ruma_summary_image(
            ruma_summary=ruma_summary,
            frame_shape=frame_shape,
            base_path=base_path,
            timestamp=timestamp,
            frame_count=frame_count,
            detection_zone=detection_zone
        )

    print(f"✅ Alerta local guardada: {alert_type} - {timestamp.strftime('%H:%M:%S')}")

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
