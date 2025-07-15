from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class RumaData:
    def __init__(self, ruma_id, initial_mask, initial_area, centroid):
        self.id = ruma_id
        self.initial_mask = initial_mask
        self.initial_area = initial_area
        self.current_area = initial_area
        self.centroid = centroid
        self.percentage = 100.0
        self.last_seen_frame = 0
        self.is_active = True
        self.frames_without_interaction = 0
        self.last_stable_percentage = 100.0
        self.label_position = (centroid[0] - 30, centroid[1] - 10)  # Posición fija del label
        self.was_interacting = False
        self.enterprise = None
        
        # Calcular radio basado en la máscara inicial
        self.radius = self._calculate_radius(initial_mask, centroid)
    
    def _calculate_radius(self, mask, centroid):
        """Calcula el radio promedio desde el centroide hasta los puntos de la máscara"""
        distances = [np.linalg.norm(np.array(centroid) - np.array(p)) for p in mask]
        return float(np.mean(distances)) if distances else 0.0