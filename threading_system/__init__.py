"""
threading_system/__init__.py

Sistema de threading para procesamiento paralelo de video.
Permite captura y procesamiento simultáneo sin bloqueos.
"""

from threading_system.frame_queue import FrameQueue, FramePacket
from threading_system.capture_worker import CaptureWorker
from threading_system.processing_worker import ProcessingWorker
from threading_system.stats_reporter import StatsReporter

__all__ = [
    'FrameQueue',
    'FramePacket',
    'CaptureWorker',
    'ProcessingWorker',
    'StatsReporter'
]