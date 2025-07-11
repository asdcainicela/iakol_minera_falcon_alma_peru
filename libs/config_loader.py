"""
Cargador y validador de configuración YAML para el sistema de monitoreo.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class ConfigLoader:
    """Cargador de configuración YAML con validación."""
    
    def __init__(self, config_path: str):
        """
        Inicializa el cargador de configuración.
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config_path = Path(config_path)
        self.config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Carga la configuración desde el archivo YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error al leer el archivo YAML: {e}")
        except Exception as e:
            raise ValueError(f"Error al cargar la configuración: {e}")
    
    def get_camera_config(self, camera_number: int) -> Dict[str, Any]:
        """
        Obtiene la configuración de una cámara específica.
        
        Args:
            camera_number: Número de la cámara
            
        Returns:
            Dict con la configuración de la cámara
            
        Raises:
            ValueError: Si la cámara no existe en la configuración
        """
        if not self.config:
            raise ValueError("Configuración no cargada")
        
        cameras = self.config.get('cameras', {})
        if camera_number not in cameras:
            available_cameras = list(cameras.keys())
            raise ValueError(
                f"Cámara {camera_number} no encontrada. "
                f"Cámaras disponibles: {available_cameras}"
            )
        
        camera_config = cameras[camera_number].copy()
        
        # Validar campos requeridos
        required_fields = ['input_video', 'output_video', 'camera_sn', 'polygons']
        for field in required_fields:
            if field not in camera_config:
                raise ValueError(f"Campo requerido '{field}' no encontrado para cámara {camera_number}")
        
        # Procesar polígonos
        camera_config['polygons'] = self._process_polygons(camera_config['polygons'])
        
        return camera_config
    
    def _process_polygons(self, polygons_data: List[List]) -> List[Tuple[int, np.ndarray]]:
        """
        Procesa los polígonos de la configuración.
        
        Args:
            polygons_data: Lista de polígonos en formato [id, [[x1,y1],[x2,y2],...]]
            
        Returns:
            Lista de tuplas (id, numpy_array)
        """
        processed_polygons = []
        
        for polygon_info in polygons_data:
            if len(polygon_info) != 2:
                raise ValueError("Cada polígono debe tener formato [id, [[x1,y1],[x2,y2],...]]")
            
            polygon_id, coordinates = polygon_info
            
            if not isinstance(coordinates, list) or len(coordinates) < 3:
                raise ValueError("Cada polígono debe tener al menos 3 puntos")
            
            # Convertir a numpy array
            try:
                polygon_array = np.array(coordinates, dtype=np.int32)
                if polygon_array.shape[1] != 2:
                    raise ValueError("Cada punto debe tener exactamente 2 coordenadas [x, y]")
            except Exception as e:
                raise ValueError(f"Error al procesar polígono {polygon_id}: {e}")
            
            processed_polygons.append((polygon_id, polygon_array))
        
        return processed_polygons
    
    def get_models_config(self) -> Dict[str, str]:
        """
        Obtiene la configuración de los modelos.
        
        Returns:
            Dict con las rutas de los modelos
        """
        if not self.config:
            raise ValueError("Configuración no cargada")
        
        models = self.config.get('models', {})
        
        # Valores por defecto
        default_models = {
            'detection': 'models/model_detection.pt',
            'segmentation': 'models/model_segmentation.pt'
        }
        
        return {**default_models, **models}
    
    def get_alerts_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración de alertas.
        
        Returns:
            Dict con la configuración de alertas
        """
        if not self.config:
            raise ValueError("Configuración no cargada")
        
        alerts = self.config.get('alerts', {})
        
        # Valores por defecto
        default_alerts = {
            'api_url': 'https://fn-alma-mina.azurewebsites.net/api/alert',
            'enterprise': 'alma',
            'save_local': True,
            'save_images': True
        }
        
        return {**default_alerts, **alerts}
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración de procesamiento.
        
        Returns:
            Dict con parámetros de procesamiento
        """
        if not self.config:
            raise ValueError("Configuración no cargada")
        
        processing = self.config.get('processing', {})
        
        # Valores por defecto
        default_processing = {
            'confidence_threshold': 0.5,
            'max_frames_without_interaction': 15,
            'candidate_ruma_confirmations': 6,
            'max_distance_ruma_matching': 50
        }
        
        return {**default_processing, **processing}
    
    def get_available_cameras(self) -> List[int]:
        """
        Obtiene la lista de cámaras disponibles.
        
        Returns:
            Lista de números de cámaras disponibles
        """
        if not self.config:
            raise ValueError("Configuración no cargada")
        
        return list(self.config.get('cameras', {}).keys())
    
    def validate_config(self) -> bool:
        """
        Valida la configuración completa.
        
        Returns:
            True si la configuración es válida
            
        Raises:
            ValueError: Si hay errores en la configuración
        """
        if not self.config:
            raise ValueError("Configuración no cargada")
        
        # Validar que existan las secciones principales
        required_sections = ['cameras']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Sección requerida '{section}' no encontrada")
        
        # Validar cada cámara
        cameras = self.config.get('cameras', {})
        if not cameras:
            raise ValueError("No se encontraron cámaras en la configuración")
        
        for camera_number in cameras:
            try:
                self.get_camera_config(camera_number)
            except ValueError as e:
                raise ValueError(f"Error en cámara {camera_number}: {e}")
        
        return True


def load_camera_config(camera_number: int, config_path: str) -> Tuple[str, str, List[Tuple[int, np.ndarray]], str, Optional[str]]:
    """
    Función de compatibilidad para cargar configuración de cámara.
    
    Args:
        camera_number: Número de la cámara
        config_path: Ruta al archivo de configuración
        
    Returns:
        Tupla con (input_video, output_video, polygons, camera_sn, save_data_dir)
        
    Raises:
        ValueError: Si hay errores en la configuración
    """
    try:
        config_loader = ConfigLoader(config_path)
        camera_config = config_loader.get_camera_config(camera_number)
        
        input_video = camera_config['input_video']
        output_video = camera_config['output_video']
        polygons = camera_config['polygons']
        camera_sn = camera_config['camera_sn']
        save_data_dir = camera_config.get('save_data')
        
        return input_video, output_video, polygons, camera_sn, save_data_dir
        
    except Exception as e:
        raise ValueError(f"Error al cargar configuración de cámara {camera_number}: {e}")