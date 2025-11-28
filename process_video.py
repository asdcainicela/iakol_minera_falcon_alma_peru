"""
process_video.py

Script principal para procesamiento multithreading con TODAS LAS OPTIMIZACIONES.
"""

import argparse
import signal
import sys
import time
import torch

from libs.config_loader import load_camera_config
from utils.paths import generar_output_video

from monitor.ruma_monitor import RumaMonitor
from threading_system import FrameQueue, CaptureWorker, ProcessingWorker, StatsReporter


workers = None


def signal_handler(sig, frame):
    """Maneja Ctrl+C de forma limpia"""
    print("\n[MAIN] Señal de interrupción recibida (Ctrl+C)")
    if workers:
        stop_all_workers(*workers)
    sys.exit(0)


def stop_all_workers(capture, processing, stats):
    """Detiene todos los workers de forma ordenada"""
    print("\n[MAIN] Deteniendo workers...")
    capture.stop(timeout=5)
    processing.stop(timeout=15)
    stats.stop(timeout=3)
    print("[MAIN] Todos los workers detenidos")


def main():
    global workers
    
    parser = argparse.ArgumentParser(description="Procesamiento OPTIMIZADO con threading")
    parser.add_argument("camera_number", type=int, help="Número de cámara")
    parser.add_argument("--queue-size", type=int, default=30, help="Tamaño de cola")
    parser.add_argument("--stats-interval", type=int, default=100, help="Intervalo de stats")
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("-"*70)
    print(f"[MAIN] Sistema OPTIMIZADO - Cámara {args.camera_number}")
    print("-"*70 + "\n")
    
    # ========== 1. CARGAR CONFIGURACIÓN (CON NUEVO PARÁMETRO) ==========
    try:
        (input_video, _, polygons, camera_sn, _, transformer, use_rtsp,
         save_video, start_video, end_video, time_save_rtsp,
         seg_idle, seg_active, cooldown, det_skip) = load_camera_config(
            args.camera_number, config_path="mkdocs.yml"
        )
        
        print(f"[MAIN] Configuración optimizada:")
        print(f"  - Fuente: {'RTSP' if use_rtsp else 'Archivo local'}")
        print(f"  - Guardar video: {'Sí' if save_video else 'No'}")
        print(f"  - Segmentación IDLE: cada {seg_idle} frames")
        print(f"  - Segmentación ACTIVE: cada {seg_active} frames")
        print(f"  - Detección SLEEP: cada {det_skip} frames")
        print(f"  - Cooldown: {cooldown} frames\n")
        
    except ValueError as e:
        print(f"[MAIN] ERROR: {e}")
        return 1
    
    # ========== 2. PREPARAR PATHS ==========
    camera_sn_clean = camera_sn.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_')
    output_video = generar_output_video(input_video, camera_sn=camera_sn_clean)
    
    model_det_path = 'models/model_detection.pt'
    model_seg_path = 'models/model_segmentation.pt'
    api_url = "https://api.ia-kol.com/api/Alert/create-alert-va"
    
    # ========== 3. CREAR COLA Y WORKERS ==========
    frame_queue = FrameQueue(maxsize=args.queue_size)
    
    capture_worker = CaptureWorker(
        video_path=input_video,
        frame_queue=frame_queue,
        use_rtsp=use_rtsp
    )
    
    if not capture_worker.start():
        print("[MAIN] ERROR: No se pudo iniciar captura")
        return 1
    
    time.sleep(1)
    video_info = capture_worker.get_video_info()
    fps = video_info.get('fps', 25.0)
    
    print(f"[MAIN] Info del video:")
    print(f"  - Resolución: {video_info.get('width')}x{video_info.get('height')}")
    print(f"  - FPS: {fps:.2f}\n")

    # ========== 4. INICIALIZAR MONITOR CON TODAS LAS OPTIMIZACIONES ==========
    print("[MAIN] Inicializando RumaMonitor OPTIMIZADO...")

    if isinstance(polygons, dict):
        detection_zone = polygons[args.camera_number]
    else:
        detection_zone = polygons

    # CRÍTICO: Llamar a process_video está en threading_system, 
    # pero el monitor se crea en ProcessingWorker
    # Necesitamos crear el monitor AQUÍ con los parámetros correctos

    from libs.main_functions import process_video

    # Preparar paths de modelos
    model_det_path = 'models/model_detection.pt'
    model_seg_path = 'models/model_segmentation.pt'

    # NO crear monitor aquí, se crea en ProcessingWorker
    # Pero necesitamos pasar los parámetros de optimización

    with torch.no_grad():
        monitor = RumaMonitor(
            model_det_path=model_det_path,
            model_seg_path=model_seg_path,
            detection_zone=detection_zone,
            camera_sn=camera_sn,
            api_url=api_url,
            transformer=transformer,
            save_video=save_video,
            segmentation_interval_idle=seg_idle,
            segmentation_interval_active=seg_active,
            activity_cooldown_frames=cooldown,
            detection_skip_idle=det_skip
        )

    print("[MAIN] RumaMonitor OPTIMIZADO inicializado\n")

    # ========== 5-7. WORKERS ==========
    processing_worker = ProcessingWorker(
        frame_queue=frame_queue,
        monitor=monitor,
        fps=fps,
        output_video_path=output_video if save_video else None
    )
    
    stats_reporter = StatsReporter(
        frame_queue=frame_queue,
        report_interval=args.stats_interval
    )
    
    workers = (capture_worker, processing_worker, stats_reporter)
    
    if not processing_worker.start():
        print("[MAIN] ERROR: No se pudo iniciar procesamiento")
        capture_worker.stop()
        return 1
    
    if not stats_reporter.start():
        print("[MAIN] ERROR: No se pudo iniciar stats")
        capture_worker.stop()
        processing_worker.stop()
        return 1
    
    print("\n[MAIN] ✅ Sistema OPTIMIZADO en ejecución")
    print("[MAIN] Presiona Ctrl+C para detener\n")
    
    # ========== 8. ESPERAR ==========
    try:
        if use_rtsp and not save_video:
            print("[MAIN] Modo RTSP continuo...")
            while True:
                time.sleep(1)
        else:
            if use_rtsp:
                duration = time_save_rtsp
                print(f"[MAIN] Grabando RTSP por {duration} segundos...")
            else:
                duration = end_video - start_video
                print(f"[MAIN] Procesando por {duration} segundos...")
            
            time.sleep(duration)
            print("\n[MAIN] Tiempo completado, deteniendo...")
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción manual (Ctrl+C)")
    
    # ========== 9. CLEANUP ==========
    stop_all_workers(*workers)
    
    print("\n" + "-"*70)
    print("[MAIN] ✅ Procesamiento OPTIMIZADO completado")
    print("-"*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())