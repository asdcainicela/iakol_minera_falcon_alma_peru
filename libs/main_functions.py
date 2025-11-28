import os
import torch
import cv2
from monitor.ruma_monitor import RumaMonitor 

def process_video(video_path, output_path, start_time_sec, end_time_sec,
                  model_det_path, model_seg_path, detection_zone, camera_number, 
                  camera_sn, api_url, transformer, use_rtsp=True, save_video=False):
    """
    Procesa un video completo usando el monitor de rumas.

    Args:
        video_path (str): Ruta del video de entrada o URL RTSP.
        output_path (str): Ruta del video de salida.
        start_time_sec (float): Tiempo de inicio en segundos.
        end_time_sec (float): Tiempo de fin en segundos (puede ser float('inf') para RTSP continuo).
        model_det_path (str): Ruta del modelo de detección.
        model_seg_path (str): Ruta del modelo de segmentación.
        detection_zone (dict[int, np.ndarray] | np.ndarray): Zonas de detección o una sola zona.
        camera_number (int): Número de la cámara.
        camera_sn (str): Número de serie de la cámara.
        api_url (str): URL de la API para enviar alertas.
        transformer: Transformador de homografía.
        use_rtsp (bool): True si es stream RTSP, False si es archivo local.
        save_video (bool): Si True, guarda el video procesado. Si False, solo procesa sin guardar.
    """
    
    # Detectar si existen modelos .engine (TensorRT)
    det_engine = model_det_path.replace('.pt', '.engine')
    seg_engine = model_seg_path.replace('.pt', '.engine')
    
    if os.path.exists(det_engine):
        print(f"[INFO] Usando modelo TensorRT para detección: {det_engine}")
        model_det_path = det_engine
    
    if os.path.exists(seg_engine):
        print(f"[INFO] Usando modelo TensorRT para segmentación: {seg_engine}")
        model_seg_path = seg_engine

    # Si detection_zone es un dict, seleccionamos la zona correspondiente
    if isinstance(detection_zone, dict):
        if camera_number not in detection_zone:
            raise ValueError(f"No hay zona definida para la cámara {camera_number}")
        detection_zone = detection_zone[camera_number]

    # Inicializar monitor con el flag de save_video
    monitor = RumaMonitor(model_det_path, model_seg_path, detection_zone, 
                         camera_sn, api_url, transformer, save_video=save_video)

    # Configurar opciones de captura según use_rtsp
    if use_rtsp:
        print("[INFO] Detectado stream RTSP, configurando opciones de captura...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    else:
        print("[INFO] Detectado video local...")
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS")

    # Para RTSP, el FPS puede ser 0 o incorrecto, usar valor por defecto
    if fps <= 0 or use_rtsp:
        fps = 25.0  # FPS por defecto para streams
        print(f"[INFO] Usando FPS por defecto: {fps}")

    # Configurar video de salida solo si save_video es True
    out = None
    if save_video:
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"[INFO] Video de salida configurado: {output_path}")
    else:
        print("[INFO] Modo sin grabación - solo procesamiento")

    # Calcular frames según use_rtsp
    if use_rtsp:
        start_frame = 0
        if save_video:
            # Si se guarda video RTSP, usar end_time_sec como duración
            end_frame = int(end_time_sec * fps)
            print(f"[INFO] Stream RTSP con grabación: grabando aproximadamente {end_frame} frames")
        else:
            # Si NO se guarda video, procesar indefinidamente
            end_frame = float('inf')
            print("[INFO] Stream RTSP sin grabación: procesamiento continuo (Ctrl+C para detener)")
    else:
        # Para archivos MP4, usar start_time_sec y end_time_sec
        start_frame = int(start_time_sec * fps)
        end_frame = int(end_time_sec * fps)
        print(f"Procesando frames {start_frame} a {end_frame}")

    frame_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 30  # Reintentar hasta 30 errores consecutivos
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Manejo de errores de lectura (importante para RTSP)
            if not ret:
                consecutive_errors += 1
                print(f"[WARN] Error leyendo frame {frame_count} (errores consecutivos: {consecutive_errors})")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("[ERROR] Demasiados errores consecutivos. Finalizando...")
                    break
                    
                # Para RTSP, intentar reconectar
                if use_rtsp:
                    print("[INFO] Intentando reconectar al stream RTSP...")
                    cap.release()
                    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        print("[ERROR] No se pudo reconectar")
                        break
                continue
            
            # Resetear contador de errores si se lee correctamente
            consecutive_errors = 0
            
            # Verificar si ya llegamos al límite
            if frame_count >= end_frame:
                print(f"[INFO] Alcanzado frame límite: {end_frame}")
                break

            if frame_count >= start_frame:
                processed_frame = monitor.process_frame(frame, frame_count, fps)
                
                # Solo escribir si save_video está activo
                if save_video and out is not None:
                    out.write(processed_frame)

                if frame_count % 50 == 0:
                    print(f"Procesados {frame_count} frames")
                    print(f"Rumas activas: {sum(1 for r in monitor.tracker.rumas.values() if r.is_active)}")

            frame_count += 1

    cap.release()
    if out is not None:
        out.release()

    if save_video:
        print(f"Procesamiento completado. Video guardado en: {output_path}")
    else:
        print(f"Procesamiento completado (sin guardar video)")
    print(f"Total de rumas detectadas: {len(monitor.tracker.rumas)}")