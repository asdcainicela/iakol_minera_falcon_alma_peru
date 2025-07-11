"""
Librería principal para el sistema de monitoreo de rumas.
"""

from .main_functions import process_video, load_camera_config
from .config_loader import ConfigLoader

__version__ = "1.0.0"
__author__ = "Sistema de Monitoreo de Rumas"

__all__ = [
    "process_video",
    "load_camera_config", 
    "ConfigLoader"
]