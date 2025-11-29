import os
import time
import torch
import cv2
from monitor.ruma_monitor import RumaMonitor
from utils.stream_monitor import StreamMonitor

def process_video(video_path, output_path, start_time_sec, end_time_sec,
                  model_det_path, model_seg_path, detection_zone, camera_number, 
                  camera_sn, api_url, transformer, use_rtsp=True, save_video=False):
    """
    Procesa un video completo usando el monitor de rumas - OPTIMIZADO PARA 30+ FPS.
    
    Sistema de contadores:
    - frames_received: Frames que llegan del stream (sin filtro)
    - frames_limited: Frames después del limitador de 6 FPS
    - frames_processed: Frames procesados completamente
    - frames_written: Frames escritos al video de salida

    Args:
        video_path (str): Ruta del video de entrada o URL RTSP.
        output_path (str): Ruta del video de salida.
        start_time_sec (float): Tiempo de inicio en segundos.
        end_time_sec (float): Tiempo de fin en segundos (para RTSP = duración de grabación en tiempo real).
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

    # Si detection_zone es un dict, seleccionamos la zona correspondiente
    if isinstance(detection_zone, dict):
        if camera_number not in detection_zone:
            raise ValueError(f"No hay zona definida para la cámara {camera_number}")
        detection_zone = detection_zone[camera_number]

    # Inicializar monitor de estadísticas del stream
    stream_monitor = StreamMonitor(
        report_interval=5.0,  # Reportar cada 5 segundos
        enable_console=True
    )

    # Inicializar monitor con el flag de save_video
    monitor = RumaMonitor(model_det_path, model_seg_path, detection_zone, 
                         camera_sn, api_url, transformer, save_video=save_video)

    # ============================================================================
    # CONFIGURACIÓN OPTIMIZADA PARA 30+ FPS
    # ============================================================================
    
    if use_rtsp:
        print("[INFO] Detectado stream RTSP, aplicando configuración OPTIMIZADA para 30+ FPS...")
        
        # ✅ CLAVE 1: Configuración OpenCV para baja latencia
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "stimeout;5000000|"
            "buffer_size;4096000|"      # 🔥 Buffer GRANDE (4MB)
            "max_delay;500000|"          # Máximo 0.5s de delay
            "reorder_queue_size;0|"      # Sin reordenamiento
            "fflags;nobuffer+fastseek|"  # Sin buffering + seek rápido
            "flags;low_delay|"
            "probesize;32768|"           # Probe pequeño
            "analyzeduration;0"          # Sin análisis previo
        )
        
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        # ✅ CLAVE 2: Buffer interno de OpenCV GRANDE
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 50)  # 🔥 50 frames de buffer
        
    else:
        print("[INFO] Detectado video local...")
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS")

    # Para RTSP o FPS inválido, usar valor por defecto
    if fps <= 0 or use_rtsp:
        fps = 25.0  # FPS por defecto para streams y video de salida
        print(f"[INFO] Usando FPS estándar para video de salida: {fps}")
    
    # Configurar video de salida solo si save_video es True
    out = None
    if save_video:
        output_fps = 25.0
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
        print(f"[INFO] Video de salida configurado: {output_path}")
        print(f"[INFO] FPS del video de salida: {output_fps}")
    else:
        print("[INFO] Modo sin grabación - solo procesamiento")

    # Calcular frames según use_rtsp
    if use_rtsp:
        start_frame = 0
        if save_video:
            end_frame = float('inf')
            print(f"[INFO] Stream RTSP: grabando durante {end_time_sec} segundos de tiempo REAL")
            print(f"[INFO] Se capturarán todos los frames que lleguen en ese tiempo")
        else:
            end_frame = float('inf')
            print("[INFO] Stream RTSP sin grabación: procesamiento continuo (Ctrl+C para detener)")
    else:
        start_frame = int(start_time_sec * fps)
        end_frame = int(end_time_sec * fps)
        print(f"Procesando frames {start_frame} a {end_frame}")

    # ============================================================================
    # CONTADORES DETALLADOS
    # ============================================================================
    frames_received = 0          # Frames recibidos del stream (sin filtro)
    frames_limited = 0           # Frames después del limitador de 6 FPS
    frames_processed = 0         # Frames procesados completamente (con YOLO)
    frames_written = 0           # Frames escritos al video
    frames_errors = 0            # Frames con error de lectura
    
    consecutive_errors = 0
    max_consecutive_errors = 30
    
    # ✅ Límite de FPS de PROCESAMIENTO ajustable
    max_processing_fps = 6.0  # Procesar 6 FPS (puedes ajustar)
    min_frame_interval = 1.0 / max_processing_fps
    last_process_time = 0.0
    
    # Control de tiempo real para RTSP con grabación
    recording_start_time = None
    max_recording_time = None
    
    if use_rtsp and save_video:
        recording_start_time = time.time()
        max_recording_time = end_time_sec
        print(f"[INFO] Iniciando grabación por {end_time_sec} segundos...")
    
    # Para calcular FPS general
    fps_calc_start_time = time.time()
    fps_calc_interval = 1.0  # Calcular FPS cada segundo
    last_fps_calc_time = fps_calc_start_time
    
    # Contadores para FPS instantáneo
    fps_received_last_second = 0
    fps_processed_last_second = 0
    
    print(f"\n{'='*80}")
    print("🚀 INICIANDO PROCESAMIENTO DE VIDEO")
    print(f"{'='*80}")
    print(f"📥 Modo de lectura: CONTINUA (30+ FPS esperado)")
    print(f"⚙️  Procesamiento pesado: máximo {max_processing_fps} FPS")
    print(f"🎯 Estrategia: Leer TODOS los frames, procesar 1 de cada {int(30/max_processing_fps)}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        while cap.isOpened():
            # Verificar timeout de grabación para RTSP
            if recording_start_time and max_recording_time:
                elapsed_real_time = time.time() - recording_start_time
                if elapsed_real_time >= max_recording_time:
                    print(f"\n[INFO] ✅ Alcanzado tiempo límite de grabación: {max_recording_time}s")
                    print(f"[INFO] Tiempo real transcurrido: {elapsed_real_time:.1f}s")
                    print(f"[INFO] Frames recibidos: {frames_received}")
                    break
            
            # ============================================================================
            # LECTURA DE FRAME (SIN FILTRO)
            # ============================================================================
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            
            # Manejo de errores de lectura
            if not ret:
                frames_errors += 1
                consecutive_errors += 1
                
                # No imprimir warning en cada error (muy verboso)
                if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                    print(f"[WARN] Error leyendo frame (errores consecutivos: {consecutive_errors})")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("[ERROR] Demasiados errores consecutivos. Finalizando...")
                    break
                    
                # Para RTSP, intentar reconectar
                if use_rtsp:
                    print("[INFO] Intentando reconectar al stream RTSP...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 50)  # Restaurar buffer
                    if not cap.isOpened():
                        print("[ERROR] No se pudo reconectar")
                        break
                continue
            
            # ✅ Frame recibido correctamente
            frames_received += 1
            fps_received_last_second += 1
            consecutive_errors = 0
            
            # Registrar en stream_monitor
            stream_monitor.frame_read()
            
            # Verificar límite (solo para MP4)
            if frames_received >= end_frame:
                print(f"[INFO] Alcanzado frame límite: {end_frame}")
                break

            # ============================================================================
            # DECIDIR SI PROCESAR ESTE FRAME (LIMITADOR DE 6 FPS)
            # ============================================================================
            current_time = time.time()
            time_since_last_process = current_time - last_process_time
            
            should_process = time_since_last_process >= min_frame_interval
            
            if should_process:
                # ✅ Frame pasa el limitador
                frames_limited += 1
                
                # Procesar solo si está en el rango
                if frames_received >= start_frame:
                    # 🔥 PROCESAMIENTO COMPLETO (pesado)
                    process_start = time.time()
                    processed_frame = monitor.process_frame(frame, frames_received, fps)
                    process_time = time.time() - process_start
                    
                    frames_processed += 1
                    fps_processed_last_second += 1
                    
                    stream_monitor.frame_processed(process_time)
                    last_process_time = current_time
                    
                    # Escribir frame procesado
                    if save_video and out is not None:
                        out.write(processed_frame)
                        frames_written += 1
                    
                    # ============================================================================
                    # LOG DETALLADO CADA 50 FRAMES PROCESADOS
                    # ============================================================================
                    if frames_processed % 50 == 0:
                        # Calcular métricas
                        elapsed_total = time.time() - fps_calc_start_time
                        fps_general = frames_received / elapsed_total if elapsed_total > 0 else 0
                        
                        drop_limitador = frames_received - frames_limited
                        drop_limitador_pct = (drop_limitador / frames_received * 100) if frames_received > 0 else 0
                        
                        drop_general = frames_received - frames_processed
                        drop_general_pct = (drop_general / frames_received * 100) if frames_received > 0 else 0
                        
                        active_rumas = sum(1 for r in monitor.tracker.rumas.values() if r.is_active)
                        active_objects = len(monitor.object_tracker.tracked_objects)
                        
                        # LOG COMPLETO
                        if recording_start_time:
                            elapsed = time.time() - recording_start_time
                            print(
                                f"[Frame {frames_received:>6}] "
                                f"⏱️ {elapsed:>5.1f}s/{max_recording_time}s | "
                                f"📥 Recibido:{frames_received:>6} | "
                                f"⚙️ Limitado:{frames_limited:>6} | "
                                f"✅ Procesado:{frames_processed:>5} | "
                                f"⏭️ Drop Limit:{drop_limitador:>5} ({drop_limitador_pct:>4.1f}%) | "
                                f"⏭️ Drop Gen:{drop_general:>5} ({drop_general_pct:>4.1f}%) | "
                                f"📊 FPS:{fps_general:>5.1f} | "
                                f"🎯 R:{active_rumas} O:{active_objects}"
                            )
                        else:
                            print(
                                f"[Frame {frames_received:>6}] "
                                f"📥 Recibido:{frames_received:>6} | "
                                f"⚙️ Limitado:{frames_limited:>6} | "
                                f"✅ Procesado:{frames_processed:>5} | "
                                f"⏭️ Drop Limit:{drop_limitador:>5} ({drop_limitador_pct:>4.1f}%) | "
                                f"⏭️ Drop Gen:{drop_general:>5} ({drop_general_pct:>4.1f}%) | "
                                f"📊 FPS:{fps_general:>5.1f} | "
                                f"⏱️ ProcTime:{process_time*1000:>5.1f}ms | "
                                f"🎯 R:{active_rumas} O:{active_objects}"
                            )
                else:
                    # Frame fuera del rango de procesamiento
                    stream_monitor.frame_skipped()
            else:
                # ❌ Frame rechazado por limitador de 6 FPS
                stream_monitor.frame_skipped()
                
                # Escribir frame original sin procesar (si save_video está activo)
                if save_video and out is not None and frames_received >= start_frame:
                    out.write(frame)
                    frames_written += 1
            
            # ============================================================================
            # CALCULAR FPS INSTANTÁNEO CADA SEGUNDO
            # ============================================================================
            if current_time - last_fps_calc_time >= fps_calc_interval:
                time_diff = current_time - last_fps_calc_time
                fps_received_instant = fps_received_last_second / time_diff
                fps_processed_instant = fps_processed_last_second / time_diff
                
                # Resetear contadores
                fps_received_last_second = 0
                fps_processed_last_second = 0
                last_fps_calc_time = current_time

    cap.release()
    if out is not None:
        out.release()

    # ============================================================================
    # REPORTE FINAL DETALLADO
    # ============================================================================
    elapsed_total = time.time() - fps_calc_start_time
    
    print("\n" + "="*80)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("="*80)
    
    if save_video:
        print(f"📹 Video guardado en: {output_path}")
        if recording_start_time:
            total_time = time.time() - recording_start_time
            print(f"⏱️  Tiempo de grabación: {total_time:.1f} segundos")
            print(f"📊 Frames recibidos del stream: {frames_received}")
            print(f"💾 Frames escritos al video: {frames_written}")
            video_duration = frames_written / 25.0
            print(f"🎬 Duración del video: ~{video_duration:.1f} segundos")
    else:
        print(f"📊 Procesamiento sin grabación completado")
    
    print(f"\n{'='*80}")
    print("📊 ESTADÍSTICAS DETALLADAS DE FRAMES")
    print(f"{'='*80}")
    print(f"📥 Frames recibidos (sin filtro):     {frames_received:>8}")
    print(f"⚙️  Frames después de limitador 6fps: {frames_limited:>8}")
    print(f"✅ Frames procesados (con YOLO):      {frames_processed:>8}")
    print(f"💾 Frames escritos al video:          {frames_written:>8}")
    print(f"❌ Frames con error de lectura:       {frames_errors:>8}")
    print()
    
    # Calcular drops
    drop_limitador = frames_received - frames_limited
    drop_limitador_pct = (drop_limitador / frames_received * 100) if frames_received > 0 else 0
    
    drop_procesamiento = frames_limited - frames_processed
    drop_procesamiento_pct = (drop_procesamiento / frames_limited * 100) if frames_limited > 0 else 0
    
    drop_general = frames_received - frames_processed
    drop_general_pct = (drop_general / frames_received * 100) if frames_received > 0 else 0
    
    print(f"⏭️  DROP por limitador de 6fps:        {drop_limitador:>8} ({drop_limitador_pct:>5.1f}%)")
    print(f"⏭️  DROP en procesamiento:             {drop_procesamiento:>8} ({drop_procesamiento_pct:>5.1f}%)")
    print(f"⏭️  DROP GENERAL (recibido→procesado): {drop_general:>8} ({drop_general_pct:>5.1f}%)")
    print()
    
    # FPS promedio
    fps_general = frames_received / elapsed_total if elapsed_total > 0 else 0
    fps_procesamiento = frames_processed / elapsed_total if elapsed_total > 0 else 0
    
    print(f"📊 FPS GENERAL (recepción):            {fps_general:>8.2f} fps")
    print(f"⚡ FPS PROCESAMIENTO (con YOLO):       {fps_procesamiento:>8.2f} fps")
    print(f"⏱️  TIEMPO TOTAL:                       {elapsed_total:>8.1f} segundos")
    
    print(f"\n{'='*80}")
    print("🎯 ESTADÍSTICAS DE DETECCIÓN")
    print(f"{'='*80}")
    print(f"🎯 Total de rumas detectadas:          {len(monitor.tracker.rumas)}")
    print(f"👥 Total de objetos trackeados:        {len(monitor.object_tracker.tracked_objects)}")
    print(f"{'='*80}\n")
    
    # Imprimir estadísticas finales del stream monitor
    stream_monitor.print_final_report()
    
    # Retornar estadísticas extendidas
    return {
        **stream_monitor.get_stats_dict(),
        'frames_received': frames_received,
        'frames_limited': frames_limited,
        'frames_processed': frames_processed,
        'frames_written': frames_written,
        'frames_errors': frames_errors,
        'drop_limitador': drop_limitador,
        'drop_limitador_pct': drop_limitador_pct,
        'drop_general': drop_general,
        'drop_general_pct': drop_general_pct,
        'fps_general': fps_general,
        'fps_procesamiento': fps_procesamiento,
        'total_rumas': len(monitor.tracker.rumas),
        'total_objects': len(monitor.object_tracker.tracked_objects)
    }