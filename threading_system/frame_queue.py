"""
threading_system/frame_queue.py

Cola thread-safe para transferir frames entre captura y procesamiento.
Incluye métricas de rendimiento para monitoreo.
"""

import queue
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FramePacket:
    """Paquete que contiene un frame y su metadata"""
    frame: np.ndarray
    frame_number: int
    capture_timestamp: float


class FrameQueue:
    """
    Cola thread-safe con métricas de rendimiento.
    
    Maneja el flujo de frames entre el thread de captura y procesamiento,
    con control de saturación y estadísticas.
    """
    
    def __init__(self, maxsize: int = 30):
        """
        Args:
            maxsize: Tamaño máximo de la cola. Si se llena, se dropean frames.
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        
        # Métricas
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_stats_time = time.time()
        
    def put(self, frame_packet: FramePacket, block: bool = False) -> bool:
        """
        Intenta agregar un frame a la cola.
        
        Args:
            frame_packet: Paquete con frame y metadata
            block: Si True, espera si la cola está llena. Si False, dropea.
            
        Returns:
            True si se agregó exitosamente, False si se dropeó
        """
        self.frames_captured += 1
        
        try:
            self.queue.put(frame_packet, block=block, timeout=0.1)
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False
    
    def get(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """
        Obtiene un frame de la cola.
        
        Args:
            timeout: Segundos a esperar si la cola está vacía
            
        Returns:
            FramePacket o None si timeout
        """
        try:
            packet = self.queue.get(timeout=timeout)
            self.frames_processed += 1
            return packet
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Retorna el tamaño actual de la cola"""
        return self.queue.qsize()
    
    def get_stats(self) -> dict:
        """
        Retorna estadísticas actuales.
        
        Returns:
            Dict con métricas de rendimiento
        """
        drop_rate = (self.frames_dropped / max(1, self.frames_captured)) * 100
        lag = self.frames_captured - self.frames_processed
        saturation = (self.size() / self.maxsize) * 100
        
        return {
            'captured': self.frames_captured,
            'processed': self.frames_processed,
            'dropped': self.frames_dropped,
            'drop_rate': drop_rate,
            'lag': lag,
            'queue_size': self.size(),
            'saturation': saturation
        }
    
    def reset_stats(self):
        """Resetea las métricas (útil para intervalos de reporte)"""
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_stats_time = time.time()