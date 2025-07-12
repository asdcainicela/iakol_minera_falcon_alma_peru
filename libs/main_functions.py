import torch
import cv2
from monitor.ruma_monitor import RumaMonitor 

def process_video(video_path, output_path, start_time_sec, end_time_sec,
                  model_det_path, model_seg_path, detection_zone, camera_number, camera_sn):
    """
    Procesa un video completo usando el monitor de rumas.

    Args:
        video_path (str): Ruta del video de entrada.
        output_path (str): Ruta del video de salida.
        start_time_sec (float): Tiempo de inicio en segundos.
        end_time_sec (float): Tiempo de fin en segundos.
        model_det_path (str): Ruta del modelo de detección.
        model_seg_path (str): Ruta del modelo de segmentación.
        detection_zone (dict[int, np.ndarray] | np.ndarray): Zonas de detección o una sola zona.
        camera_number (int): Número de la cámara.
        camera_sn (str): Número de serie de la cámara.
    """

    # Si detection_zone es un dict, seleccionamos la zona correspondiente
    if isinstance(detection_zone, dict):
        if camera_number not in detection_zone:
            raise ValueError(f"No hay zona definida para la cámara {camera_number}")
        detection_zone = detection_zone[camera_number]

    # Inicializar monitor
    monitor = RumaMonitor(model_det_path, model_seg_path, detection_zone, camera_sn)

    # Configurar video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

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
                processed_frame = monitor.process_frame(frame, frame_count, fps)
                out.write(processed_frame)

                if frame_count % 50 == 0:
                    print(f"Procesados {frame_count} frames")
                    print(f"Rumas activas: {sum(1 for r in monitor.tracker.rumas.values() if r.is_active)}")

            frame_count += 1

    cap.release()
    out.release()

    print(f"Procesamiento completado. Video guardado en: {output_path}")
    print(f"Total de rumas detectadas: {len(monitor.tracker.rumas)}")
