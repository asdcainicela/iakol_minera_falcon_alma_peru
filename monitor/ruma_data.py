import cv2
import numpy as np

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
