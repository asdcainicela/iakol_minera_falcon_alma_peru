from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

@dataclass
class RumaInfo:
    id: int
    percent: float
    centroid: Tuple[int, int]
    radius: float
    centroid_homographic: Optional[Tuple[float, float]] = None
    radius_homographic: Optional[float] = None

@dataclass
class AlertContext:
    frame: np.ndarray
    frame_count: int
    fps: float
    camera_sn: str
    enterprise: str = "default"
    ruma_summary: Optional[dict] = None
    frame_shape: Optional[Tuple[int, int]] = None
    detection_zone: Optional[List[Tuple[int, int]]] = None
