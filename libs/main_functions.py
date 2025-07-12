import torch
import cv2
import yaml
import numpy as np
from monitor.ruma_monitor import RumaMonitor
from monitor.ruma_tracker import RumaTracker
from monitor.ruma_data import RumaData
from utils.geometry import is_point_in_polygon, calculate_intersection
from alerts.alert_manager import save_alert
from utils.paths import setup_alerts_folder
from utils.draw import put_text_with_background, draw_zone_and_status

from libs.config_loader import load_camera_config

def process_video(video_path, output_path, start_time_sec, end_time_sec,
                 model_det_path, model_seg_path, detection_zone, camera_number, camera_sn):
    """
    Procesa un video completo usando el monitor de rumas

    Args:
        video_path: Ruta del video de entrada
        output_path: Ruta del video de salida
        start_time_sec: Tiempo de inicio en segundos
        end_time_sec: Tiempo de fin en segundos
        model_det_path: Ruta del modelo de detección
        model_seg_path: Ruta del modelo de segmentación
        detection_zone: Zona de detección como array numpy o dict
        camera_number: Número de cámara (1, 2, o 3)
        camera_sn: Serial de la cámara (string)
    """

    # ✅ Ya NO se debe hacer get() porque camera_sn es un string directamente
    # Inicializar monitor
    monitor = RumaMonitor(model_det_path, model_seg_path, detection_zone, camera_sn)

    # Configurar video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f} FPS")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)

    print(f"Procesando frames {start_frame} a {end_frame}")

    frame_count = 0

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > end_frame:
                break

            if frame_count >= start_frame:
                # Procesar frame
                processed_frame = monitor.process_frame(frame, frame_count, fps)
                out.write(processed_frame)

                if frame_count % 50 == 0:
                    print(f"Procesados {frame_count} frames")
                    print(f"Rumas activas: {sum(1 for r in monitor.tracker.rumas.values() if r.is_active)}")

            frame_count += 1

    cap.release()
    out.release()
    print(f"Procesamiento completado. Video guardado en {output_path}")
    print(f"Total de rumas detectadas: {len(monitor.tracker.rumas)}")
