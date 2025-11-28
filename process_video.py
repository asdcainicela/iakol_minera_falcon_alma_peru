"""
process_video_threaded.py

Script principal para procesamiento multithreading.
NO toca process_video.py original - es una alternativa paralela.

Uso:
    python process_video_threaded.py 1    # Procesar cámara 1
    python process_video_threaded.py 2    # Procesar cámara 2
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


# Variable global para manejar señales de interrupción
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
    
    # Detener captura primero (deja de agregar frames)
    capture.stop(timeout=5)
    
    # Esperar a que processing termine los frames restantes
    processing.stop(timeout=15)
    
    # Detener stats (reporte final)
    stats.stop(timeout=3)
    
    print("[MAIN] Todos los workers detenidos")


def main():
    global workers
    
    parser = argparse.ArgumentParser(description="Procesamiento de video con threading")
    parser.add_argument("camera_number", type=int, help="Número de cámara (mkdocs.yml)")
    parser.add_argument("--queue-size", type=int, default=30, 
                       help="Tamaño de la cola de frames (default: 30)")
    parser.add_argument("--stats-interval", type=int, default=100,
                       help="Intervalo de reporte de stats (default: 100 frames)")
    
    args = parser.parse_args()
    
    # Registrar handler para Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("-"*70)
    print(f"[MAIN] Iniciando procesamiento threaded - Cámara {args.camera_number}")
    print("-"*70 + "\n")
    
    # ========== 1. CARGAR CONFIGURACIÓN ==========
    try:
        (input_video, _, polygons, camera_sn, _, transformer, use_rtsp,
         save_video, start_video, end_video, time_save_rtsp) = load_camera_config(
            args.camera_number, config_path="mkdocs.yml"
        )
        
        print(f"[MAIN] Configuración cargada:")
        print(f"  - Fuente: {'RTSP' if use_rtsp else 'Archivo local'}")
        print(f"  - Guardar video: {'Sí' if save_video else 'No'}")
        print(f"  - Cámara SN: {camera_sn}\n")
        
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
    
    # Worker de captura
    capture_worker = CaptureWorker(
        video_path=input_video,
        frame_queue=frame_queue,
        use_rtsp=use_rtsp
    )
    
    # Iniciar captura para obtener info del video
    if not capture_worker.start():
        print("[MAIN] ERROR: No se pudo iniciar captura")
        return 1
    
    time.sleep(1)  # Esperar a que capture lea primer frame
    video_info = capture_worker.get_video_info()
    fps = video_info.get('fps', 25.0)
    
    print(f"[MAIN] Info del video:")
    print(f"  - Resolución: {video_info.get('width')}x{video_info.get('height')}")
    print(f"  - FPS: {fps:.2f}\n")
    
    # ========== 4. INICIALIZAR MONITOR (tu código existente) ==========
    print("[MAIN] Inicializando RumaMonitor...")
    
    if isinstance(polygons, dict):
        detection_zone = polygons[args.camera_number]
    else:
        detection_zone = polygons
    
    with torch.no_grad():
        monitor = RumaMonitor(
            model_det_path=model_det_path,
            model_seg_path=model_seg_path,
            detection_zone=detection_zone,
            camera_sn=camera_sn,
            api_url=api_url,
            transformer=transformer,
            save_video=save_video
        )
    
    print("[MAIN] RumaMonitor inicializado\n")
    
    # ========== 5. WORKER DE PROCESAMIENTO ==========
    processing_worker = ProcessingWorker(
        frame_queue=frame_queue,
        monitor=monitor,
        fps=fps,
        output_video_path=output_video if save_video else None
    )
    
    # ========== 6. WORKER DE ESTADÍSTICAS ==========
    stats_reporter = StatsReporter(
        frame_queue=frame_queue,
        report_interval=args.stats_interval
    )
    
    # ========== 7. INICIAR TODOS LOS WORKERS ==========
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
    
    print("\n[MAIN] Todos los workers iniciados")
    print("[MAIN] Presiona Ctrl+C para detener\n")
    
    # ========== 8. ESPERAR (CTRL+C PARA DETENER) ==========
    try:
        # Para RTSP sin grabación, corre indefinidamente
        if use_rtsp and not save_video:
            print("[MAIN] Modo RTSP continuo - procesando indefinidamente...")
            while True:
                time.sleep(1)
        
        # Para RTSP con grabación o MP4, esperar el tiempo especificado
        else:
            if use_rtsp:
                duration = time_save_rtsp
                print(f"[MAIN] Grabando RTSP por {duration} segundos...")
            else:
                duration = end_video - start_video
                print(f"[MAIN] Procesando video por {duration} segundos...")
            
            time.sleep(duration)
            print("\n[MAIN] Tiempo completado, deteniendo...")
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción manual (Ctrl+C)")
    
    # ========== 9. CLEANUP ==========
    stop_all_workers(*workers)
    
    print("\n" + "-"*70)
    print("[MAIN] Procesamiento completado exitosamente")
    print("-"*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())