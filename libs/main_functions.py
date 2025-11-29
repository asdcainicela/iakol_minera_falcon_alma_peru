import os
import time
import torch
import cv2
from monitor.ruma_monitor import RumaMonitor
from utils.stream_monitor import StreamMonitor
from utils.video_capture_thread import VideoCaptureThread

def process_video(video_path, output_path, start_time_sec, end_time_sec,
                  model_det_path, model_seg_path, detection_zone, camera_number, 
                  camera_sn, api_url, transformer, use_rtsp=True, save_video=False):
    """
    Procesa un video usando THREAD DEDICADO para captura continua.
    
    ARQUITECTURA:
    - Thread 1 (capture): Lee frames del stream a 30+ FPS continuamente
    - Thread 2 (main): Procesa frames a 6 FPS con YOLO
    
    Esto evita pérdida de frames durante procesamiento pesado.

    Args:
        video_path (str): Ruta del video de entrada o URL RTSP.
        output_path (str): Ruta del video de salida.
        start_time_sec (float): Tiempo de inicio en segundos.
        end_time_sec (float): Tiempo de fin en segundos.
        model_det_path (str): Ruta del modelo de detección.
        model_seg_path (str): Ruta del modelo de segmentación.
        detection_zone (dict[int, np.ndarray] | np.ndarray): Zonas de detección.
        camera_number (int): Número de la cámara.
        camera_sn (str): Número de serie de la cámara.
        api_url (str): URL de la API para enviar alertas.
        transformer: Transformador de homografía.
        use_rtsp (bool): True si es stream RTSP, False si es archivo local.
        save_video (bool): Si True, guarda el video procesado.
    """

    # Si detection_zone es un dict, seleccionamos la zona correspondiente
    if isinstance(detection_zone, dict):
        if camera_number not in detection_zone:
            raise ValueError(f"No hay zona definida para la cámara {camera_number}")
        detection_zone = detection_zone[camera_number]

    # Inicializar monitor de estadísticas
    stream_monitor = StreamMonitor(
        report_interval=5.0,
        enable_console=True
    )

    # Inicializar monitor de rumas
    monitor = RumaMonitor(model_det_path, model_seg_path, detection_zone, 
                         camera_sn, api_url, transformer, save_video=save_video)

    # ============================================================================
    # INICIALIZAR CAPTURA CON THREAD DEDICADO
    # ============================================================================
    
    print("\n" + "="*80)
    print("🚀 INICIALIZANDO SISTEMA DE CAPTURA CON THREAD DEDICADO")
    print("="*80)
    print(f"📹 Fuente: {video_path}")
    print(f"🧵 Thread de captura: buffer de 100 frames")
    print(f"⚙️  Thread de procesamiento: máximo 6 FPS")
    print("="*80 + "\n")
    
    # Crear thread de captura
    capture = VideoCaptureThread(
        video_source=video_path,
        buffer_size=100,  # Buffer grande para manejar picos
        use_rtsp=use_rtsp
    )
    
    # Iniciar captura
    capture.start()
    
    # Esperar a que se llene el buffer inicial
    print("[INFO] Esperando buffer inicial...")
    time.sleep(2)
    
    # Obtener info del video
    video_info = capture.get_video_info()
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    
    print(f"[INFO] Video: {width}x{height} @ {fps:.2f} FPS")

    # Configurar video de salida
    out = None
    if save_video:
        output_fps = 25.0
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                             output_fps, (width, height))
        print(f"[INFO] Video de salida: {output_path} @ {output_fps} FPS")
    else:
        print("[INFO] Modo sin grabación - solo procesamiento")

    # Calcular límites de frames
    if use_rtsp:
        start_frame = 0
        if save_video:
            end_frame = float('inf')
            print(f"[INFO] Stream RTSP: grabando {end_time_sec}s de tiempo real")
        else:
            end_frame = float('inf')
            print("[INFO] Stream RTSP: procesamiento continuo (Ctrl+C para detener)")
    else:
        start_frame = int(start_time_sec * fps)
        end_frame = int(end_time_sec * fps)
        print(f"[INFO] Video local: frames {start_frame} a {end_frame}")

    # ============================================================================
    # CONTADORES DETALLADOS
    # ============================================================================
    frames_received = 0      # Frames leídos del thread de captura
    frames_limited = 0       # Frames después del limitador de 6 FPS
    frames_processed = 0     # Frames procesados con YOLO
    frames_written = 0       # Frames escritos al video
    frames_read_errors = 0   # Errores al leer del buffer
    
    # Límite de FPS de PROCESAMIENTO
    max_processing_fps = 6.0
    min_frame_interval = 1.0 / max_processing_fps
    last_process_time = 0.0
    
    # Control de tiempo real para RTSP
    recording_start_time = None
    max_recording_time = None
    
    if use_rtsp and save_video:
        recording_start_time = time.time()
        max_recording_time = end_time_sec
        print(f"[INFO] Tiempo de grabación: {end_time_sec}s")
    
    # Para calcular FPS
    fps_calc_start_time = time.time()
    
    print(f"\n{'='*80}")
    print("🎬 INICIANDO PROCESAMIENTO")
    print(f"{'='*80}")
    print(f"📥 Thread captura: leyendo a máxima velocidad (~30 FPS)")
    print(f"⚡ Thread proceso: procesando a 6 FPS máximo")
    print(f"{'='*80}\n")
    
    try:
        with torch.no_grad():
            while True:
                # Verificar timeout de grabación
                if recording_start_time and max_recording_time:
                    elapsed_real_time = time.time() - recording_start_time
                    if elapsed_real_time >= max_recording_time:
                        print(f"\n[INFO] ✅ Tiempo límite alcanzado: {max_recording_time}s")
                        print(f"[INFO] Tiempo transcurrido: {elapsed_real_time:.1f}s")
                        print(f"[INFO] Frames recibidos: {frames_received}")
                        break
                
                # ============================================================================
                # LEER FRAME DEL THREAD DE CAPTURA
                # ============================================================================
                ret, frame = capture.read(timeout=1.0)
                
                if not ret:
                    frames_read_errors += 1
                    
                    # Si hay muchos errores consecutivos, verificar thread
                    if frames_read_errors > 50:
                        if not capture.is_running:
                            print("[ERROR] Thread de captura se detuvo")
                            break
                    
                    continue
                
                # ✅ Frame recibido del thread
                frames_received += 1
                stream_monitor.frame_read()
                
                # Verificar límite (solo para MP4)
                if frames_received >= end_frame:
                    print(f"[INFO] Frame límite alcanzado: {end_frame}")
                    break

                # ============================================================================
                # LIMITADOR DE PROCESAMIENTO (6 FPS)
                # ============================================================================
                current_time = time.time()
                time_since_last_process = current_time - last_process_time
                
                should_process = time_since_last_process >= min_frame_interval
                
                if should_process:
                    # ✅ Frame pasa el limitador
                    frames_limited += 1
                    
                    if frames_received >= start_frame:
                        # 🔥 PROCESAMIENTO COMPLETO (YOLO + tracking)
                        process_start = time.time()
                        processed_frame = monitor.process_frame(frame, frames_received, fps)
                        process_time = time.time() - process_start
                        
                        frames_processed += 1
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
                            # Obtener estadísticas del thread de captura
                            capture_stats = capture.get_stats()
                            
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
                                    f"⏭️ Drop:{drop_general:>5} ({drop_general_pct:>4.1f}%) | "
                                    f"📊 FPS:{fps_general:>5.1f} | "
                                    f"🧵 Captura:{capture_stats['capture_fps']:>5.1f} | "
                                    f"📦 Buf:{capture_stats['buffer_size']:>3}/{capture_stats['buffer_max']} | "
                                    f"🎯 R:{active_rumas} O:{active_objects}"
                                )
                            else:
                                print(
                                    f"[Frame {frames_received:>6}] "
                                    f"📥 Recibido:{frames_received:>6} | "
                                    f"✅ Procesado:{frames_processed:>5} | "
                                    f"⏭️ Drop:{drop_general:>5} ({drop_general_pct:>4.1f}%) | "
                                    f"📊 FPS:{fps_general:>5.1f} | "
                                    f"🧵 Captura:{capture_stats['capture_fps']:>5.1f} | "
                                    f"📦 Buf:{capture_stats['buffer_size']:>3}/{capture_stats['buffer_max']} | "
                                    f"⏱️ ProcTime:{process_time*1000:>5.1f}ms | "
                                    f"🎯 R:{active_rumas} O:{active_objects}"
                                )
                    else:
                        stream_monitor.frame_skipped()
                else:
                    # ❌ Frame rechazado por limitador
                    stream_monitor.frame_skipped()
                    
                    # Escribir frame sin procesar
                    if save_video and out is not None and frames_received >= start_frame:
                        out.write(frame)
                        frames_written += 1
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por usuario (Ctrl+C)")
    
    finally:
        # ============================================================================
        # LIMPIEZA Y CIERRE
        # ============================================================================
        print("\n[INFO] Deteniendo captura...")
        capture.stop()
        
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
            print(f"📊 Frames recibidos del thread: {frames_received}")
            print(f"💾 Frames escritos al video: {frames_written}")
            video_duration = frames_written / 25.0
            print(f"🎬 Duración del video: ~{video_duration:.1f} segundos")
    else:
        print(f"📊 Procesamiento sin grabación completado")
    
    print(f"\n{'='*80}")
    print("📊 ESTADÍSTICAS DETALLADAS DE FRAMES")
    print(f"{'='*80}")
    print(f"📥 Frames recibidos del thread:       {frames_received:>8}")
    print(f"⚙️  Frames después de limitador 6fps: {frames_limited:>8}")
    print(f"✅ Frames procesados (con YOLO):      {frames_processed:>8}")
    print(f"💾 Frames escritos al video:          {frames_written:>8}")
    print(f"❌ Errores al leer del buffer:        {frames_read_errors:>8}")
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
    
    # Estadísticas del thread de captura
    print(f"\n{'='*80}")
    capture.print_stats()
    
    print(f"{'='*80}")
    print("🎯 ESTADÍSTICAS DE DETECCIÓN")
    print(f"{'='*80}")
    print(f"🎯 Total de rumas detectadas:          {len(monitor.tracker.rumas)}")
    print(f"👥 Total de objetos trackeados:        {len(monitor.object_tracker.tracked_objects)}")
    print(f"{'='*80}\n")
    
    # Imprimir estadísticas finales del stream monitor
    stream_monitor.print_final_report()
    
    # Retornar estadísticas completas
    capture_stats = capture.get_stats()
    
    return {
        **stream_monitor.get_stats_dict(),
        'frames_received': frames_received,
        'frames_limited': frames_limited,
        'frames_processed': frames_processed,
        'frames_written': frames_written,
        'frames_read_errors': frames_read_errors,
        'drop_limitador': drop_limitador,
        'drop_limitador_pct': drop_limitador_pct,
        'drop_general': drop_general,
        'drop_general_pct': drop_general_pct,
        'fps_general': fps_general,
        'fps_procesamiento': fps_procesamiento,
        'capture_fps': capture_stats['capture_fps'],
        'capture_dropped': capture_stats['frames_dropped'],
        'total_rumas': len(monitor.tracker.rumas),
        'total_objects': len(monitor.object_tracker.tracked_objects)
    }