"""
This module initializes the library for video processing and configuration loading.
"""

from .main_functions import process_video 
from .config_loader import load_camera_config

__version__ = "1.0.0"
__author__ = "Sistema de Monitoreo de Rumas"

__all__ = [
    "process_video",
    "load_camera_config"
]