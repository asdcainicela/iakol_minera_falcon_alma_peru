"""
threading_system/stats_reporter.py

Thread que reporta estadísticas periódicamente.
Cada N frames imprime métricas de rendimiento sin bloquear los otros threads.
"""

import time
import threading
from typing import Optional

from threading_system.frame_queue import FrameQueue


class StatsReporter:
    """
    Worker que imprime estadísticas periódicamente.
    
    Reporta: frames capturados, procesados, drop rate, latencia, etc.
    """
    
    def __init__(self, frame_queue: FrameQueue, report_interval: int = 100):
        """
        Args:
            frame_queue: Cola para obtener estadísticas
            report_interval: Cada cuántos frames capturados reportar
        """
        self.frame_queue = frame_queue
        self.report_interval = report_interval
        
        # Control del thread
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Para calcular frames/sec
        self.last_report_time = time.time()
        self.last_captured = 0
        self.last_processed = 0
    
    def _stats_loop(self):
        """Loop principal de reportes (corre en thread separado)"""
        print("[StatsReporter] Iniciando reportes cada", self.report_interval, "frames...")
        
        self.last_report_time = time.time()
        
        while not self.stop_event.is_set():
            time.sleep(2)  # Chequear cada 2 segundos
            
            stats = self.frame_queue.get_stats()
            
            # Reportar solo si alcanzamos el intervalo
            if stats['captured'] - self.last_captured >= self.report_interval:
                self._print_report(stats)
                
                # Actualizar contadores
                self.last_captured = stats['captured']
                self.last_processed = stats['processed']
                self.last_report_time = time.time()
        
        print("[StatsReporter] Thread de estadísticas finalizado")
    
    def _print_report(self, stats: dict):
        """Imprime un reporte formateado"""
        elapsed = time.time() - self.last_report_time
        
        # Calcular throughput
        captured_delta = stats['captured'] - self.last_captured
        processed_delta = stats['processed'] - self.last_processed
        
        capture_fps = captured_delta / elapsed if elapsed > 0 else 0
        process_fps = processed_delta / elapsed if elapsed > 0 else 0
        
        print("\n" + "-"*70)
        print(f"[STATS] Frame #{stats['captured']}")
        print("-"*70)
        print(f"  Capturados:     {stats['captured']:6d}  ({capture_fps:5.1f} fps)")
        print(f"  Procesados:     {stats['processed']:6d}  ({process_fps:5.1f} fps)")
        print(f"  Dropeados:      {stats['dropped']:6d}")
        print(f"  Drop Rate:      {stats['drop_rate']:6.2f}%")
        print(f"  Lag (frames):   {stats['lag']:6d}")
        print(f"  Queue Size:     {stats['queue_size']:3d}/{self.frame_queue.maxsize} "
              f"({stats['saturation']:5.1f}% llena)")
        print("-"*70 + "\n")
        
        # Warnings
        if stats['drop_rate'] > 10:
            print("⚠️  WARNING: Drop rate > 10% - Considerar reducir carga de procesamiento")
        
        if stats['saturation'] > 80:
            print("⚠️  WARNING: Queue > 80% llena - Procesamiento muy lento")
    
    def start(self) -> bool:
        """
        Inicia el thread de estadísticas.
        
        Returns:
            True si se inició exitosamente
        """
        if self.is_running:
            print("[StatsReporter] Ya está corriendo")
            return False
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._stats_loop, daemon=True)
        self.thread.start()
        self.is_running = True
        
        print("[StatsReporter] Thread iniciado")
        return True
    
    def stop(self, timeout: float = 3.0):
        """
        Detiene el thread de estadísticas.
        
        Args:
            timeout: Segundos a esperar antes de forzar cierre
        """
        if not self.is_running:
            return
        
        print("[StatsReporter] Señal de stop enviada...")
        self.stop_event.set()
        
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        
        # Reporte final
        final_stats = self.frame_queue.get_stats()
        print("\n" + "-"*70)
        print("[STATS] REPORTE FINAL")
        print("-"*70)
        print(f"  Total Capturados:  {final_stats['captured']}")
        print(f"  Total Procesados:  {final_stats['processed']}")
        print(f"  Total Dropeados:   {final_stats['dropped']}")
        print(f"  Drop Rate Final:   {final_stats['drop_rate']:.2f}%")
        print("-"*70 + "\n")
        
        self.is_running = False
        print("[StatsReporter] Thread detenido")