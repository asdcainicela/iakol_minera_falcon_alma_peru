import os
import torch
import cv2
from monitor.ruma_monitor import RumaMonitor 

def process_video(video_path, output_path, start_time_sec, end_time_sec,
                  model_det_path, model_seg_path, detection_zone, camera_number, 
                  camera_sn, api_url, transformer, use_rtsp=True, save_video=False,
                  segmentation_interval_idle=100,
                  segmentation_interval_active=10,
                  activity_cooldown_frames=200,
                  detection_skip_idle=3):
    """
    Procesa un video completo usando el monitor de rumas.
    """
    
    # ========== DETECCIÓN DE MODELOS TENSORRT (MEJORADO) ==========
    print(f"\n[INFO] Buscando modelos...")
    print(f"  - Detección .pt:  {model_det_path}")
    print(f"  - Segmentación .pt: {model_seg_path}")
    
    det_engine = model_det_path.replace('.pt', '.engine')
    seg_engine = model_seg_path.replace('.pt', '.engine')
    
    print(f"\n[INFO] Verificando modelos TensorRT...")
    print(f"  - Detección .engine: {det_engine}")
    print(f"    Existe: {os.path.exists(det_engine)}")
    print(f"  - Segmentación .engine: {seg_engine}")
    print(f"    Existe: {os.path.exists(seg_engine)}")
    
    # CRÍTICO: Reemplazar paths si existen .engine
    if os.path.exists(det_engine):
        print(f"[INFO] ✅ Usando TensorRT para detección")
        model_det_path = det_engine
    else:
        print(f"[WARN] ⚠️  NO se encontró {det_engine}, usando PyTorch (LENTO)")
    
    if os.path.exists(seg_engine):
        print(f"[INFO] ✅ Usando TensorRT para segmentación")
        model_seg_path = seg_engine
    else:
        print(f"[WARN] ⚠️  NO se encontró {seg_engine}, usando PyTorch (LENTO)")
    
    print(f"\n[INFO] Modelos finales a cargar:")
    print(f"  - Detección: {model_det_path}")
    print(f"  - Segmentación: {model_seg_path}\n")
    # ================================================================

    # Si detection_zone es un dict, seleccionamos la zona correspondiente
    if isinstance(detection_zone, dict):
        if camera_number not in detection_zone:
            raise ValueError(f"No hay zona definida para la cámara {camera_number}")
        detection_zone = detection_zone[camera_number]

    # Inicializar monitor CON LOS PATHS CORRECTOS y parámetros de optimización
    monitor = RumaMonitor(
        model_det_path,  # 👈 AHORA SÍ PASA EL PATH CORRECTO
        model_seg_path,  # 👈 AHORA SÍ PASA EL PATH CORRECTO
        detection_zone, 
        camera_sn, 
        api_url, 
        transformer, 
        save_video=save_video,
        segmentation_interval_idle=segmentation_interval_idle,
        segmentation_interval_active=segmentation_interval_active,
        activity_cooldown_frames=activity_cooldown_frames,
        detection_skip_idle=detection_skip_idle
    )

    # ... resto del código sin cambios (desde "Configurar opciones de captura" hasta el final)
    
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
        fps = 25.0
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
            end_frame = int(end_time_sec * fps)
            print(f"[INFO] Stream RTSP con grabación: grabando aproximadamente {end_frame} frames")
        else:
            end_frame = float('inf')
            print("[INFO] Stream RTSP sin grabación: procesamiento continuo (Ctrl+C para detener)")
    else:
        start_frame = int(start_time_sec * fps)
        end_frame = int(end_time_sec * fps)
        print(f"Procesando frames {start_frame} a {end_frame}")

    frame_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 30
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                consecutive_errors += 1
                print(f"[WARN] Error leyendo frame {frame_count} (errores consecutivos: {consecutive_errors})")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("[ERROR] Demasiados errores consecutivos. Finalizando...")
                    break
                    
                if use_rtsp:
                    print("[INFO] Intentando reconectar al stream RTSP...")
                    cap.release()
                    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        print("[ERROR] No se pudo reconectar")
                        break
                continue
            
            consecutive_errors = 0
            
            if frame_count >= end_frame:
                print(f"[INFO] Alcanzado frame límite: {end_frame}")
                break

            if frame_count >= start_frame:
                processed_frame = monitor.process_frame(frame, frame_count, fps)
                
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