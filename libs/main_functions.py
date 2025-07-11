"""
Funciones principales para el procesamiento de video y monitoreo de rumas.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .config_loader import ConfigLoader
from monitor.ruma_monitor import RumaMonitor
from utils.paths import generar_ruta_salida


def process_video(
    camera_number: int,
    input_path: str,
    output_path: str,
    model_detection: str,
    model_segmentation: str,
    polygons: List[Tuple[int, np.ndarray]],
    camera_sn: str,
    save_data_dir: Optional[str] = None,
    start_time_sec: float = 0,
    end_time_sec: Optional[float] = None,
    show_display: bool = False,
    show_drawings: bool = True,
    save_video: bool = True,
    save_data: bool = True,
    config_loader: Optional[ConfigLoader] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Procesa un video completo usando el monitor de rumas.
    
    Args:
        camera_number: Número de la cámara
        input_path: Ruta del video de entrada o stream RTSP
        output_path: Ruta del video de salida
        model_detection: Ruta del modelo de detección
        model_segmentation: Ruta del modelo de segmentación
        polygons: Lista de polígonos de detección [(id, np.array)]
        camera_sn: Número de serie de la cámara
        save_data_dir: Directorio para guardar datos
        start_time_sec: Tiempo de inicio en segundos
        end_time_sec: Tiempo de fin en segundos (None para todo el video)
        show_display: Mostrar video en tiempo real
        show_drawings: Mostrar dibujos en el video
        save_video: Guardar video procesado
        save_data: Guardar datos de alertas
        config_loader: Instancia del cargador de configuración
        
    Returns:
        Tupla con (transitions, avg_times, id_history)
    """
    
    # Obtener configuración si está disponible
    processing_config = {}
    alerts_config = {}
    
    if config_loader:
        try:
            processing_config = config_loader.get_processing_config()
            alerts_config = config_loader.get_alerts_config()
        except Exception as e:
            print(f"Advertencia: No se pudo cargar configuración adicional: {e}")
    
    # Usar el primer polígono como zona de detección principal
    if not polygons:
        raise ValueError("Se requiere al menos un polígono de detección")
    
    detection_zone = polygons[0][1]  # Tomar el array numpy del primer polígono
    
    # Inicializar monitor con configuración
    monitor = RumaMonitor(
        model_det_path=model_detection,
        model_seg_path=model_segmentation,
        detection_zone=detection_zone,
        camera_sn=camera_sn,
        processing_config=processing_config,
        alerts_config=alerts_config,
        save_data_dir=save_data_dir
    )
    
    # Configurar captura de video
    if input_path.startswith('rtsp://') or input_path.startswith('http://'):
        # Stream en vivo
        cap = cv2.VideoCapture(input_path)
        is_stream = True
    else:
        # Archivo de video
        cap = cv2.VideoCapture(input_path)
        is_stream = False
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {input_path}")
    
    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0:
        fps = 30  # Valor por defecto para streams
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print(f"Fuente: {'Stream' if is_stream else 'Archivo'}")
    
    # Configurar escritor de video si es necesario
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calcular frames de inicio y fin
    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps) if end_time_sec else None
    
    if not is_stream:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames
        print(f"Procesando frames {start_frame} a {end_frame} de {total_frames}")
    else:
        print(f"Procesando stream desde {start_time_sec}s")
    
    frame_count = 0
    processed_frames = 0
    
    # Variables para estadísticas
    transitions = {}
    avg_times = {}
    id_history = {}
    
    try:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if is_stream:
                        print("Reconectando stream...")
                        cap.release()
                        cap = cv2.VideoCapture(input_path)
                        if not cap.isOpened():
                            break
                        continue
                    else:
                        break
                
                # Verificar si hemos llegado al final del rango
                if end_frame and frame_count > end_frame:
                    break
                
                # Procesar frame si está en el rango
                if frame_count >= start_frame:
                    # Procesar frame con el monitor
                    processed_frame = monitor.process_frame(frame, frame_count, fps)
                    
                    # Guardar video si es necesario
                    if save_video and out:
                        out.write(processed_frame)
                    
                    # Mostrar video si es necesario
                    if show_display:
                        cv2.imshow('Monitoreo de Rumas', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    processed_frames += 1
                    
                    # Mostrar progreso
                    if processed_frames % 100 == 0:
                        print(f"Procesados {processed_frames} frames")
                        rumas_activas = sum(1 for r in monitor.rumas.values() if r.is_active)
                        print(f"Rumas activas: {rumas_activas}")
                
                frame_count += 1
                
                # Para streams, evitar procesar indefinidamente
                if is_stream and end_time_sec and (frame_count / fps) > end_time_sec:
                    break
    
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")
    
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        raise
    
    finally:
        # Limpiar recursos
        cap.release()
        if out:
            out.release()
        if show_display:
            cv2.destroyAllWindows()
    
    # Generar estadísticas finales
    print(f"\nProcesamiento completado:")
    print(f"- Frames procesados: {processed_frames}")
    print(f"- Total de rumas detectadas: {len(monitor.rumas)}")
    print(f"- Rumas activas: {sum(1 for r in monitor.rumas.values() if r.is_active)}")
    
    if save_video:
        print(f"- Video guardado en: {output_path}")
    
    return transitions, avg_times, id_history


def process_video_from_config(
    camera_number: int,
    config_path: str,
    start_time_sec: float = 0,
    end_time_sec: Optional[float] = None,
    show_display: bool = False,
    save_video: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Procesa un video usando configuración YAML.
    
    Args:
        camera_number: Número de la cámara
        config_path: Ruta al archivo de configuración
        start_time_sec: Tiempo de inicio en segundos
        end_time_sec: Tiempo de fin en segundos
        show_display: Mostrar video en tiempo real
        save_video: Guardar video procesado
        
    Returns:
        Tupla con (transitions, avg_times, id_history)
    """
    
    # Cargar configuración
    config_loader = ConfigLoader(config_path)
    camera_config = config_loader.get_camera_config(camera_number)
    models_config = config_loader.get_models_config()
    
    # Extraer parámetros
    input_video = camera_config['input_video']
    output_video = camera_config['output_video']
    polygons = camera_config['polygons']
    camera_sn = camera_config['camera_sn']
    save_data_dir = camera_config.get('save_data')
    
    # Generar ruta de salida automática si no se especifica
    if not output_video or output_video == "auto":
        output_video = generar_ruta_salida(input_video)
    
    model_detection = models_config['detection']
    model_segmentation = models_config['segmentation']
    
    # Validar que existan los modelos
    if not Path(model_detection).exists():
        raise FileNotFoundError(f"Modelo de detección no encontrado: {model_detection}")
    
    if not Path(model_segmentation).exists():
        raise FileNotFoundError(f"Modelo de segmentación no encontrado: {model_segmentation}")
    
    # Procesar video
    return process_video(
        camera_number=camera_number,
        input_path=input_video,
        output_path=output_video,
        model_detection=model_detection,
        model_segmentation=model_segmentation,
        polygons=polygons,
        camera_sn=camera_sn,
        save_data_dir=save_data_dir,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        show_display=show_display,
        save_video=save_video,
        config_loader=config_loader
    )


def validate_camera_setup(camera_number: int, config_path: str) -> bool:
    """
    Valida la configuración de una cámara específica.
    
    Args:
        camera_number: Número de la cámara
        config_path: Ruta al archivo de configuración
        
    Returns:
        True si la configuración es válida
        
    Raises:
        ValueError: Si hay errores en la configuración
    """
    
    config_loader = ConfigLoader(config_path)
    camera_config = config_loader.get_camera_config(camera_number)
    models_config = config_loader.get_models_config()
    
    # Validar video de entrada
    input_video = camera_config['input_video']
    if not (input_video.startswith('rtsp://') or input_video.startswith('http://') or Path(input_video).exists()):
        raise ValueError(f"Video de entrada no válido: {input_video}")
    
    # Validar modelos
    model_detection = models_config['detection']
    model_segmentation = models_config['segmentation']
    
    if not Path(model_detection).exists():
        raise ValueError(f"Modelo de detección no encontrado: {model_detection}")
    
    if not Path(model_segmentation).exists():
        raise ValueError(f"Modelo de segmentación no encontrado: {model_segmentation}")
    
    # Validar polígonos
    polygons = camera_config['polygons']
    if not polygons:
        raise ValueError("Se requiere al menos un polígono de detección")
    
    print(f"Configuración de cámara {camera_number} validada correctamente")
    return True


# Función de compatibilidad con el código anterior
def load_camera_config(camera_number: int, config_path: str) -> Tuple[str, str, List[Tuple[int, np.ndarray]], str, Optional[str]]:
    """
    Función de compatibilidad para cargar configuración de cámara.
    
    Args:
        camera_number: Número de la cámara
        config_path: Ruta al archivo de configuración
        
    Returns:
        Tupla con (input_video, output_video, polygons, camera_sn, save_data_dir)
    """
    from .config_loader import load_camera_config as _load_camera_config
    return _load_camera_config(camera_number, config_path)