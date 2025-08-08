import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

from alerts.alert_info import RumaInfo, AlertContext
from alerts.alert_utils import save_ruma_summary_image, save_ruma_summary_image_homography

from utils.paths import generar_folder_fecha

def save_alert_local(
    alert_type: str,
    ruma_data: RumaInfo = None,
    context: AlertContext = None
):
    """Guarda localmente una alerta con imagen y metadatos"""
    timestamp = datetime.now()
    #base_path = generar_folder_fecha("alerts_save", etiqueta="local")
    #print(" Save alert local ejecutándose")
    camera_id = int(context.camera_sn.split('-')[-1])

    if camera_id == 1:
        ruta_img = "ref/homography/img_map/Mapa1_nuevo.png"
        base_path = generar_folder_fecha("alerts_save", etiqueta="cam1_local")
    elif camera_id == 2:
        ruta_img = "ref/homography/img_map/Mapa2_nuevo.png"
        base_path = generar_folder_fecha("alerts_save", etiqueta="cam2_local")
    elif camera_id == 3:
        ruta_img = "ref/homography/img_map/Mapa3_nuevo.png"
        base_path = generar_folder_fecha("alerts_save", etiqueta="cam3_local")
    else:
        ruta_img = "ref/homography/img_map/Mapa1_nuevo.png"
        base_path = generar_folder_fecha("alerts_save", etiqueta="cam1_local")

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
                base_path=base_path,
                timestamp=timestamp,
                frame_count=context.frame_count, #context.detection_zone,
                map_image_path=ruta_img
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
    #with open(json_path, 'w') as f:
    #    json.dump(metadata, f, indent=2)
    #cv2.imwrite(str(image_path), context.frame)
    
    # Guardar JSON
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Guardar imagen solo si el frame es válido
    if context.frame is not None and context.frame.size != 0:
        cv2.imwrite(str(image_path), context.frame)
    else:
        print(f"[Error] El frame está vacío. No se pudo guardar la imagen en: {image_path}")


    #print(f" Alerta local guardada: {alert_type} - {timestamp.strftime('%H:%M:%S')}")

