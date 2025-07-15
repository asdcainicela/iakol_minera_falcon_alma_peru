import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class RumaData:
    id: int
    initial_mask: np.ndarray
    initial_area: float
    centroid: Tuple[int, int]

    # Campos con valores por defecto
    current_area: float = field(init=False)
    percentage: float = field(default=100.0)
    last_seen_frame: int = field(default=0)
    is_active: bool = field(default=True)
    frames_without_interaction: int = field(default=0)
    last_stable_percentage: float = field(default=100.0)
    label_position: Tuple[int, int] = field(init=False)
    was_interacting: bool = field(default=False)
    enterprise: Optional[str] = field(default=None)
    radius: float = field(init=False)

    def __post_init__(self):
        self.current_area = self.initial_area
        self.label_position = (self.centroid[0] - 30, self.centroid[1] - 10)
        self.radius = self._calculate_radius(self.initial_mask, self.centroid)

    def _calculate_radius(self, mask: np.ndarray, centroid: Tuple[int, int]) -> float:
        """Calcula el radio promedio desde el centroide hasta los puntos de la máscara"""
        distances = [np.linalg.norm(np.array(centroid) - np.array(p)) for p in mask]
        return float(np.mean(distances)) if distances else 0.0
