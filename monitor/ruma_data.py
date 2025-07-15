from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np

@dataclass
class RumaData:
    id: int
    initial_mask: np.ndarray
    initial_area: float
    centroid: Tuple[int, int]

    current_area: float = field(init=False)
    percentage: float = field(default=100.0)
    last_seen_frame: int = field(default=0)
    is_active: bool = field(default=True)
    frames_without_interaction: int = field(default=0)
    last_stable_percentage: float = field(default=100.0)
    label_position: Tuple[int, int] = field(init=False)
    was_interacting: bool = field(default=False)
    enterprise: Optional[str] = field(default=None)

    def __post_init__(self):
        self.current_area = self.initial_area
        self.label_position = (self.centroid[0] - 30, self.centroid[1] - 10)
