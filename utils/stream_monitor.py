"""
stream_monitor.py

Monitor para medir el FPS real de streams RTSP y detectar frames perdidos.
"""

import time
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class StreamStats:
    """Estadísticas del stream de video"""
    # Contadores de frames
    stream_frames_available: int = 0  # Frames que ofrece el stream
    frames_read: int = 0              # Frames leídos del buffer
    frames_processed: int = 0         # Frames procesados completamente
    frames_skipped: int = 0           # Frames saltados (no procesados)
    
    # Tiempos
    last_frame_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    
    # Para calcular FPS del stream
    frame_intervals: deque = field(default_factory=lambda: deque(maxlen=60))
    
    # Tiempos de procesamiento
    process_times: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def update_frame_read(self):
        """Registra un frame leído del stream"""
        current_time = time.time()
        
        # Calcular intervalo entre frames
        if self.last_frame_time > 0:
            interval = current_time - self.last_frame_time
            self.frame_intervals.append(interval)
        
        self.frames_read += 1
        self.last_frame_time = current_time
    
    def update_frame_processed(self, process_time: float):
        """Registra un frame procesado"""
        self.frames_processed += 1
        self.process_times.append(process_time)
    
    def update_frame_skipped(self):
        """Registra un frame saltado"""
        self.frames_skipped += 1
    
    @property
    def stream_fps(self) -> float:
        """FPS real del stream basado en intervalos medidos"""
        if len(self.frame_intervals) < 2:
            return 0.0
        avg_interval = sum(self.frame_intervals) / len(self.frame_intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    @property
    def processing_fps(self) -> float:
        """FPS de procesamiento"""
        if len(self.process_times) < 2:
            return 0.0
        return len(self.process_times) / sum(self.process_times)
    
    @property
    def avg_process_time_ms(self) -> float:
        """Tiempo promedio de procesamiento en ms"""
        if not self.process_times:
            return 0.0
        return (sum(self.process_times) / len(self.process_times)) * 1000
    
    @property
    def skip_rate(self) -> float:
        """Porcentaje de frames saltados"""
        if self.frames_read == 0:
            return 0.0
        return (self.frames_skipped / self.frames_read) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Tiempo transcurrido desde el inicio"""
        return time.time() - self.start_time
    
    def get_report(self) -> str:
        """Genera reporte de estadísticas del stream"""
        return (
            f"\n{'='*70}\n"
            f"📊 ESTADÍSTICAS DEL STREAM\n"
            f"{'='*70}\n"
            f"  📹 FPS del Stream:      {self.stream_fps:>7.2f} fps  (FPS real de la cámara)\n"
            f"  ⚡ FPS Procesamiento:   {self.processing_fps:>7.2f} fps  (FPS de tu sistema)\n"
            f"  📊 Eficiencia:          {(self.processing_fps/self.stream_fps*100) if self.stream_fps > 0 else 0:>6.1f}%   (% de frames procesados)\n"
            f"\n"
            f"  📥 Frames Leídos:       {self.frames_read:>8}  (del buffer)\n"
            f"  ✅ Frames Procesados:   {self.frames_processed:>8}  (completamente)\n"
            f"  ⏭️  Frames Saltados:     {self.frames_skipped:>8}  ({self.skip_rate:.1f}%)\n"
            f"\n"
            f"  ⏱️  Tiempo por Frame:    {self.avg_process_time_ms:>6.1f} ms\n"
            f"  ⏱️  Tiempo Transcurrido: {self.elapsed_time:>6.1f} s\n"
            f"{'='*70}\n"
        )


class StreamMonitor:
    """Monitor de rendimiento para streams de video"""
    
    def __init__(self, report_interval: float = 5.0, enable_console: bool = True):
        """
        Args:
            report_interval: Intervalo en segundos para reportar estadísticas
            enable_console: Si True, imprime estadísticas en consola
        """
        self.stats = StreamStats()
        self.report_interval = report_interval
        self.enable_console = enable_console
        self.last_report_time = time.time()
    
    def frame_read(self):
        """Registra que se leyó un frame del stream"""
        self.stats.update_frame_read()
    
    def frame_processed(self, process_time: float):
        """Registra que se procesó un frame"""
        self.stats.update_frame_processed(process_time)
        
        # Reportar si es tiempo
        if self.enable_console and self._should_report():
            print(self.stats.get_report(), flush=True)
    
    def frame_skipped(self):
        """Registra que se saltó un frame"""
        self.stats.update_frame_skipped()
    
    def _should_report(self) -> bool:
        """Verifica si debe reportar estadísticas"""
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.last_report_time = current_time
            return True
        return False
    
    def get_stats_dict(self) -> dict:
        """Retorna las estadísticas como diccionario"""
        return {
            'stream_fps': self.stats.stream_fps,
            'processing_fps': self.stats.processing_fps,
            'efficiency_percent': (self.stats.processing_fps / self.stats.stream_fps * 100) 
                                 if self.stats.stream_fps > 0 else 0,
            'frames_read': self.stats.frames_read,
            'frames_processed': self.stats.frames_processed,
            'frames_skipped': self.stats.frames_skipped,
            'skip_rate': self.stats.skip_rate,
            'avg_process_time_ms': self.stats.avg_process_time_ms,
            'elapsed_time_sec': self.stats.elapsed_time
        }
    
    def print_final_report(self):
        """Imprime reporte final"""
        print("\n" + "="*70)
        print("🏁 REPORTE FINAL DEL STREAM")
        print("="*70)
        print(f"  ⏱️  Tiempo Total:         {self.stats.elapsed_time:.2f} segundos")
        print(f"  📹 FPS del Stream:       {self.stats.stream_fps:.2f} fps")
        print(f"  ⚡ FPS Procesamiento:    {self.stats.processing_fps:.2f} fps")
        print(f"  📊 Eficiencia:           {(self.stats.processing_fps/self.stats.stream_fps*100) if self.stats.stream_fps > 0 else 0:.1f}%")
        print()
        print(f"  📥 Frames Leídos:        {self.stats.frames_read}")
        print(f"  ✅ Frames Procesados:    {self.stats.frames_processed}")
        print(f"  ⏭️  Frames Saltados:      {self.stats.frames_skipped} ({self.stats.skip_rate:.1f}%)")
        print()
        
        if self.stats.stream_fps > 0:
            theoretical_frames = int(self.stats.elapsed_time * self.stats.stream_fps)
            print(f"  📊 Frames Esperados:     {theoretical_frames} (basado en FPS del stream)")
            print(f"  📊 Frames Perdidos:      {theoretical_frames - self.stats.frames_read}")
        
        print("="*70 + "\n")
        
        # Advertencias y recomendaciones
        if self.stats.processing_fps < self.stats.stream_fps * 0.5:
            print("⚠️  ADVERTENCIA: El sistema está procesando menos del 50% de los frames")
            print("   Recomendaciones:")
            print("   - Considerar usar un modelo más ligero")
            print("   - Reducir la resolución de entrada")
            print("   - Implementar procesamiento paralelo")
            print("   - Procesar 1 de cada N frames\n")


if __name__ == "__main__":
    # Ejemplo de uso
    import random
    
    monitor = StreamMonitor(report_interval=2.0)
    
    print("Simulando stream de video a 25 FPS...")
    print("Procesamiento: 5 FPS (simulado)\n")
    
    frame_interval = 1.0 / 25.0  # 25 FPS
    process_time = 0.2  # 200ms por frame = 5 FPS
    
    start = time.time()
    frames_to_simulate = 100
    
    for i in range(frames_to_simulate):
        # Simular llegada de frame
        monitor.frame_read()
        
        # Simular procesamiento selectivo (1 de cada 5 frames)
        if i % 5 == 0:
            time.sleep(process_time)
            monitor.frame_processed(process_time)
        else:
            monitor.frame_skipped()
        
        # Simular intervalo entre frames del stream
        time.sleep(frame_interval)
    
    monitor.print_final_report()