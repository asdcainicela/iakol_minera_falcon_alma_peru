"""
threading_system/processing_worker.py

Thread que procesa frames usando RumaMonitor.
Consume frames de la cola y los pasa al monitor existente (sin modificarlo).
"""

import cv2
import time
import threading
from pathlib import Path
from typing import Optional

from threading_system.frame_queue import FrameQueue
from monitor.ruma_monitor import RumaMonitor


class ProcessingWorker:
    """
    Worker que procesa frames usando el RumaMonitor existente.
    
    IMPORTANTE: NO modifica RumaMonitor, solo lo llama.
    """
    
    def __init__(self, frame_queue: FrameQueue, monitor: RumaMonitor, 
                 fps: float, output_video_path: Optional[Path] = None):
        """
        Args:
            frame_queue: Cola de donde sacar frames
            monitor: Instancia de RumaMonitor (ya inicializada)
            fps: FPS del video (para cálculos de tiempo)
            output_video_path: Ruta para guardar video procesado (opcional)
        """
        self.frame_queue = frame_queue
        self.monitor = monitor
        self.fps = fps
        self.output_video_path = output_video_path
        
        # Control del thread
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Video writer (si se guarda video)
        self.video_writer: Optional[cv2.VideoWriter] = None
        
    def _init_video_writer(self, frame_shape):
        """Inicializa el VideoWriter si es necesario"""
        if not self.monitor.save_video or self.output_video_path is None:
            return
        
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.output_video_path), 
            fourcc, 
            self.fps, 
            (width, height)
        )
        
        if self.video_writer.isOpened():
            print(f"[ProcessingWorker] Video de salida: {self.output_video_path}")
        else:
            print(f"[ProcessingWorker] ERROR: No se pudo crear video de salida")
            self.video_writer = None
    
    def _processing_loop(self):
        """Loop principal de procesamiento (corre en thread separado)"""
        print("[ProcessingWorker] Iniciando loop de procesamiento...")
        
        video_writer_initialized = False
        frames_processed = 0
        
        while not self.stop_event.is_set():
            # Sacar frame de la cola
            packet = self.frame_queue.get(timeout=1.0)
            
            if packet is None:
                continue  # Timeout, reintentar
            
            # Inicializar video writer en el primer frame
            if not video_writer_initialized:
                self._init_video_writer(packet.frame.shape)
                video_writer_initialized = True
            
            # LLAMAR AL MONITOR (tu código existente, sin cambios)
            try:
                processed_frame = self.monitor.process_frame(
                    frame=packet.frame,
                    frame_count=packet.frame_number,
                    fps=self.fps
                )
                
                # Guardar video procesado si corresponde
                if self.video_writer is not None and processed_frame is not None:
                    self.video_writer.write(processed_frame)
                
                frames_processed += 1
                
                # Log cada 50 frames
                if frames_processed % 50 == 0:
                    latency = (time.time() - packet.capture_timestamp) * 1000
                    print(f"[ProcessingWorker] Procesados {frames_processed} frames | "
                          f"Latencia: {latency:.0f}ms")
                
            except Exception as e:
                print(f"[ProcessingWorker] ERROR procesando frame {packet.frame_number}: {e}")
        
        print(f"[ProcessingWorker] Loop finalizado. Total procesados: {frames_processed}")
    
    def start(self) -> bool:
        """
        Inicia el thread de procesamiento.
        
        Returns:
            True si se inició exitosamente
        """
        if self.is_running:
            print("[ProcessingWorker] Ya está corriendo")
            return False
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        self.is_running = True
        
        print("[ProcessingWorker] Thread iniciado")
        return True
    
    def stop(self, timeout: float = 10.0):
        """
        Detiene el thread de procesamiento de forma limpia.
        
        Args:
            timeout: Segundos a esperar antes de forzar cierre
        """
        if not self.is_running:
            return
        
        print("[ProcessingWorker] Señal de stop enviada...")
        self.stop_event.set()
        
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        
        # Cerrar video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("[ProcessingWorker] Video guardado y cerrado")
        
        self.is_running = False
        print("[ProcessingWorker] Thread detenido")