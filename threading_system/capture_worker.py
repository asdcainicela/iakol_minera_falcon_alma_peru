"""
threading_system/capture_worker.py

Thread dedicado a capturar frames de RTSP/video sin bloqueos.
Su único trabajo es leer frames y meterlos en la cola lo más rápido posible.
"""

import cv2
import time
import threading
from pathlib import Path
from typing import Optional

from threading_system.frame_queue import FrameQueue, FramePacket


class CaptureWorker:
    """
    Worker que captura frames de forma continua en un thread separado.
    
    Optimizado para RTSP: maneja reconexiones y timeouts automáticamente.
    """
    
    def __init__(self, video_path: str, frame_queue: FrameQueue, 
                 use_rtsp: bool = False, max_reconnect_attempts: int = 5):
        """
        Args:
            video_path: Ruta del video o URL RTSP
            frame_queue: Cola donde meter los frames capturados
            use_rtsp: Si es True, configura opciones para RTSP
            max_reconnect_attempts: Intentos de reconexión antes de fallar
        """
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.use_rtsp = use_rtsp
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Control del thread
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        
        # Estado
        self.frame_number = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        
    def _open_capture(self) -> bool:
        """
        Abre la captura de video/RTSP.
        
        Returns:
            True si se abrió exitosamente
        """
        if self.use_rtsp:
            # Configuración específica para RTSP
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
            self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"[CaptureWorker] ERROR: No se pudo abrir {self.video_path}")
            return False
        
        print(f"[CaptureWorker] Captura abierta exitosamente")
        return True
    
    def _reconnect(self) -> bool:
        """
        Intenta reconectar al stream RTSP.
        
        Returns:
            True si logró reconectar
        """
        print("[CaptureWorker] Intentando reconectar...")
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        time.sleep(2)  # Esperar antes de reintentar
        return self._open_capture()
    
    def _capture_loop(self):
        """Loop principal de captura (corre en thread separado)"""
        consecutive_errors = 0
        
        print("[CaptureWorker] Iniciando loop de captura...")
        
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_errors += 1
                print(f"[CaptureWorker] Error leyendo frame (errores consecutivos: {consecutive_errors})")
                
                # Si hay demasiados errores, intentar reconectar
                if consecutive_errors >= 10:
                    if not self._reconnect():
                        print("[CaptureWorker] Fallo en reconexión, abortando...")
                        break
                    consecutive_errors = 0
                
                time.sleep(0.1)
                continue
            
            # Frame leído exitosamente
            consecutive_errors = 0
            self.frame_number += 1
            
            # Crear paquete
            packet = FramePacket(
                frame=frame,
                frame_number=self.frame_number,
                capture_timestamp=time.time()
            )
            
            # Intentar meter en la cola (no bloqueante)
            success = self.frame_queue.put(packet, block=False)
            
            if not success and self.frame_number % 50 == 0:
                print(f"[CaptureWorker] Cola saturada, dropeando frames (frame {self.frame_number})")
        
        print("[CaptureWorker] Loop de captura finalizado")
    
    def start(self) -> bool:
        """
        Inicia el thread de captura.
        
        Returns:
            True si se inició exitosamente
        """
        if self.is_running:
            print("[CaptureWorker] Ya está corriendo")
            return False
        
        if not self._open_capture():
            return False
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.is_running = True
        
        print("[CaptureWorker] Thread iniciado")
        return True
    
    def stop(self, timeout: float = 5.0):
        """
        Detiene el thread de captura de forma limpia.
        
        Args:
            timeout: Segundos a esperar antes de forzar cierre
        """
        if not self.is_running:
            return
        
        print("[CaptureWorker] Señal de stop enviada...")
        self.stop_event.set()
        
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.is_running = False
        print("[CaptureWorker] Thread detenido")
    
    def get_video_info(self) -> dict:
        """Retorna información del video (FPS, resolución, etc.)"""
        if self.cap is None or not self.cap.isOpened():
            return {}
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or self.use_rtsp:
            fps = 25.0  # Default para RTSP
        
        return {
            'fps': fps,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }