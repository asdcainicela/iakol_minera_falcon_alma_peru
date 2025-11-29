"""
video_capture_thread.py

Thread dedicado para captura continua de frames de stream RTSP.
Evita pérdida de frames durante procesamiento pesado.
"""

import cv2
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np


class VideoCaptureThread:
    """
    Captura frames de video en un thread separado para evitar drops.
    
    Beneficios:
    - Lectura continua sin bloqueos por procesamiento
    - Buffer configurable para manejar picos de latencia
    - Estadísticas en tiempo real de captura
    """
    
    def __init__(self, video_source: str, buffer_size: int = 100, use_rtsp: bool = True):
        """
        Args:
            video_source: Ruta al video o URL RTSP
            buffer_size: Tamaño del buffer de frames (default: 100)
            use_rtsp: Si True, aplica configuraciones optimizadas para RTSP
        """
        self.video_source = video_source
        self.buffer_size = buffer_size
        self.use_rtsp = use_rtsp
        
        # Queue thread-safe para frames
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Control del thread
        self.capture_thread = None
        self.stop_flag = threading.Event()
        self.is_running = False
        
        # Estadísticas
        self.frames_captured = 0
        self.frames_dropped = 0
        self.capture_errors = 0
        self.start_time = None
        
        # VideoCapture (se inicializa en start())
        self.cap = None
        
        # Info del video
        self.width = 0
        self.height = 0
        self.fps = 0
        
    def _setup_capture(self):
        """Configura el VideoCapture con opciones optimizadas"""
        if self.use_rtsp:
            import os
            # Configuración optimizada para RTSP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "stimeout;5000000|"
                "buffer_size;8192000|"       # 8MB de buffer
                "max_delay;500000|"
                "reorder_queue_size;0|"
                "fflags;nobuffer+fastseek|"
                "flags;low_delay|"
                "probesize;32768|"
                "analyzeduration;0"
            )
            
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            
            # Buffer interno grande
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        else:
            self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            raise IOError(f"No se pudo abrir el video: {self.video_source}")
        
        # Obtener propiedades
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.fps <= 0 or self.use_rtsp:
            self.fps = 25.0  # Default para streams
        
        print(f"[VideoCaptureThread] Configurado: {self.width}x{self.height} @ {self.fps} FPS")
        print(f"[VideoCaptureThread] Buffer size: {self.buffer_size} frames")
    
    def _capture_loop(self):
        """Loop principal de captura (corre en thread separado)"""
        consecutive_errors = 0
        max_consecutive_errors = 30
        
        while not self.stop_flag.is_set():
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.capture_errors += 1
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("[VideoCaptureThread] Demasiados errores, deteniendo captura")
                        break
                    
                    # Intentar reconectar para RTSP
                    if self.use_rtsp and consecutive_errors % 10 == 0:
                        print(f"[VideoCaptureThread] Intentando reconectar... (error {consecutive_errors})")
                        self.cap.release()
                        time.sleep(2)
                        self._setup_capture()
                    
                    continue
                
                # Frame capturado exitosamente
                consecutive_errors = 0
                self.frames_captured += 1
                
                # Intentar agregar al queue (sin bloquear)
                try:
                    # Si el queue está lleno, remover el frame más viejo
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                    
                    # Agregar nuevo frame
                    self.frame_queue.put(frame, block=False)
                    
                except queue.Full:
                    self.frames_dropped += 1
                
            except Exception as e:
                print(f"[VideoCaptureThread] Error en captura: {e}")
                self.capture_errors += 1
                time.sleep(0.1)
        
        print("[VideoCaptureThread] Loop de captura finalizado")
    
    def start(self):
        """Inicia el thread de captura"""
        if self.is_running:
            print("[VideoCaptureThread] Ya está corriendo")
            return
        
        print("[VideoCaptureThread] Iniciando thread de captura...")
        
        # Configurar captura
        self._setup_capture()
        
        # Resetear flags y estadísticas
        self.stop_flag.clear()
        self.frames_captured = 0
        self.frames_dropped = 0
        self.capture_errors = 0
        self.start_time = time.time()
        
        # Iniciar thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.is_running = True
        
        print("[VideoCaptureThread] Thread iniciado correctamente")
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee un frame del buffer.
        
        Args:
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            (ret, frame): ret=True si hay frame disponible
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Detiene el thread de captura"""
        if not self.is_running:
            return
        
        print("[VideoCaptureThread] Deteniendo thread...")
        self.stop_flag.set()
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5.0)
        
        if self.cap is not None:
            self.cap.release()
        
        self.is_running = False
        print("[VideoCaptureThread] Thread detenido")
    
    def get_stats(self) -> dict:
        """Retorna estadísticas de captura"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        capture_fps = self.frames_captured / elapsed if elapsed > 0 else 0
        
        return {
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'capture_errors': self.capture_errors,
            'capture_fps': capture_fps,
            'buffer_size': self.frame_queue.qsize(),
            'buffer_max': self.buffer_size,
            'elapsed_time': elapsed
        }
    
    def print_stats(self):
        """Imprime estadísticas de captura"""
        stats = self.get_stats()
        
        drop_rate = (stats['frames_dropped'] / stats['frames_captured'] * 100) if stats['frames_captured'] > 0 else 0
        buffer_usage = (stats['buffer_size'] / stats['buffer_max'] * 100)
        
        print(f"\n{'='*70}")
        print("📹 ESTADÍSTICAS DE CAPTURA (Thread)")
        print(f"{'='*70}")
        print(f"  📥 Frames capturados:    {stats['frames_captured']:>8}")
        print(f"  ⏭️  Frames descartados:   {stats['frames_dropped']:>8} ({drop_rate:>5.1f}%)")
        print(f"  ❌ Errores de captura:   {stats['capture_errors']:>8}")
        print(f"  📊 FPS de captura:       {stats['capture_fps']:>8.2f} fps")
        print(f"  📦 Buffer actual:        {stats['buffer_size']:>8} / {stats['buffer_max']} ({buffer_usage:>5.1f}%)")
        print(f"  ⏱️  Tiempo transcurrido:  {stats['elapsed_time']:>8.1f} s")
        print(f"{'='*70}\n")
    
    def get_video_info(self) -> dict:
        """Retorna información del video"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps
        }
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()


if __name__ == "__main__":
    # Ejemplo de uso
    rtsp_url = "rtsp://admin:Facil.12@192.168.1.3:554/Streaming/Channels/101"
    
    print("Iniciando captura con thread...")
    
    with VideoCaptureThread(rtsp_url, buffer_size=100, use_rtsp=True) as capture:
        # Esperar a que se llene un poco el buffer
        time.sleep(1)
        
        # Simular procesamiento
        frames_processed = 0
        start = time.time()
        
        while time.time() - start < 10:  # 10 segundos de prueba
            ret, frame = capture.read(timeout=1.0)
            
            if ret:
                frames_processed += 1
                
                # Simular procesamiento pesado
                time.sleep(0.2)  # 200ms = 5 FPS de procesamiento
                
                if frames_processed % 10 == 0:
                    stats = capture.get_stats()
                    print(f"Procesados: {frames_processed} | "
                          f"Buffer: {stats['buffer_size']}/{stats['buffer_max']} | "
                          f"FPS captura: {stats['capture_fps']:.1f}")
        
        # Estadísticas finales
        capture.print_stats()
        print(f"\nFrames procesados: {frames_processed}")