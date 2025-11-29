"""
frame_stats_monitor.py

Sistema de monitoreo de estadísticas de frames procesados.
Incluye métricas de rendimiento, FPS, y tasa de pérdida de frames.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FrameStats:
    """Estadísticas de frames procesados"""
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    
    # Para calcular FPS
    process_times: deque = field(default_factory=lambda: deque(maxlen=30))
    last_report_time: float = field(default_factory=time.time)
    
    # Tiempos de procesamiento
    total_process_time: float = 0.0
    avg_process_time: float = 0.0
    
    def update_received(self):
        """Incrementa contador de frames recibidos"""
        self.frames_received += 1
    
    def update_processed(self, process_time: float):
        """Actualiza estadísticas de frame procesado"""
        self.frames_processed += 1
        self.process_times.append(process_time)
        self.total_process_time += process_time
        self.avg_process_time = sum(self.process_times) / len(self.process_times)
    
    def update_dropped(self):
        """Incrementa contador de frames perdidos"""
        self.frames_dropped += 1
    
    @property
    def drop_rate(self) -> float:
        """Calcula el porcentaje de frames perdidos"""
        if self.frames_received == 0:
            return 0.0
        return (self.frames_dropped / self.frames_received) * 100
    
    @property
    def processing_fps(self) -> float:
        """Calcula FPS de procesamiento actual"""
        if len(self.process_times) < 2:
            return 0.0
        return len(self.process_times) / sum(self.process_times)
    
    def should_report(self, interval: float = 5.0) -> bool:
        """Verifica si debe reportar estadísticas"""
        current_time = time.time()
        if current_time - self.last_report_time >= interval:
            self.last_report_time = current_time
            return True
        return False
    
    def get_report(self) -> str:
        """Genera reporte de estadísticas"""
        return (
            f"\n{'='*60}\n"
            f"📊 ESTADÍSTICAS DE FRAMES\n"
            f"{'='*60}\n"
            f"  📥 Frames Recibidos:    {self.frames_received:>8}\n"
            f"  ✅ Frames Procesados:   {self.frames_processed:>8}\n"
            f"  ❌ Frames Perdidos:     {self.frames_dropped:>8}\n"
            f"  📉 Tasa de Pérdida:     {self.drop_rate:>7.2f}%\n"
            f"  ⚡ FPS Procesamiento:   {self.processing_fps:>7.2f}\n"
            f"  ⏱️  Tiempo Promedio:     {self.avg_process_time*1000:>6.1f} ms\n"
            f"{'='*60}\n"
        )


class FrameStatsMonitor:
    """Monitor de estadísticas de frames con visualización"""
    
    def __init__(self, report_interval: float = 5.0, enable_console: bool = True):
        """
        Args:
            report_interval: Intervalo en segundos para reportar estadísticas
            enable_console: Si True, imprime estadísticas en consola
        """
        self.stats = FrameStats()
        self.report_interval = report_interval
        self.enable_console = enable_console
        self.start_time = time.time()
    
    def frame_received(self):
        """Registra un frame recibido"""
        self.stats.update_received()
    
    def frame_processed(self, process_time: float):
        """Registra un frame procesado con su tiempo"""
        self.stats.update_processed(process_time)
        
        # Reportar si es tiempo
        if self.enable_console and self.stats.should_report(self.report_interval):
            print(self.stats.get_report(), flush=True)
    
    def frame_dropped(self):
        """Registra un frame perdido"""
        self.stats.update_dropped()
    
    def get_stats_dict(self) -> dict:
        """Retorna las estadísticas como diccionario"""
        elapsed_time = time.time() - self.start_time
        return {
            'frames_received': self.stats.frames_received,
            'frames_processed': self.stats.frames_processed,
            'frames_dropped': self.stats.frames_dropped,
            'drop_rate': self.stats.drop_rate,
            'processing_fps': self.stats.processing_fps,
            'avg_process_time_ms': self.stats.avg_process_time * 1000,
            'elapsed_time_sec': elapsed_time
        }
    
    def print_final_report(self):
        """Imprime reporte final al terminar el procesamiento"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🏁 REPORTE FINAL DE PROCESAMIENTO")
        print("="*60)
        print(f"  ⏱️  Tiempo Total:         {elapsed:.2f} segundos")
        print(f"  📥 Frames Recibidos:    {self.stats.frames_received}")
        print(f"  ✅ Frames Procesados:   {self.stats.frames_processed}")
        print(f"  ❌ Frames Perdidos:     {self.stats.frames_dropped}")
        print(f"  📉 Tasa de Pérdida:     {self.stats.drop_rate:.2f}%")
        
        if elapsed > 0:
            print(f"  ⚡ FPS Promedio:        {self.stats.frames_processed/elapsed:.2f}")
        
        if self.stats.avg_process_time > 0:
            print(f"  ⏱️  Tiempo por Frame:    {self.stats.avg_process_time*1000:.1f} ms")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    import random
    
    monitor = FrameStatsMonitor(report_interval=2.0)
    
    print("Simulando procesamiento de video...")
    
    for i in range(100):
        monitor.frame_received()
        
        # Simular procesamiento con posibilidad de drop
        if random.random() > 0.1:  # 90% success rate
            process_time = random.uniform(0.02, 0.05)  # 20-50ms
            time.sleep(process_time)
            monitor.frame_processed(process_time)
        else:
            monitor.frame_dropped()
        
        time.sleep(0.01)  # Simular intervalo entre frames
    
    monitor.print_final_report()