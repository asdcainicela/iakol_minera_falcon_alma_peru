import cv2
import numpy as np
from monitor.ruma_data import RumaData

class RumaTracker:
    def __init__(self):
        self.rumas = {}  # ruma_id: RumaData
        self.candidate_rumas = {}  # clave: str → datos temporales
        self.next_ruma_id = 1
        self.ruma_summary = {}
        self.new_ruma_created = None

    def find_closest_ruma(self, centroid, max_distance=50):
        min_distance = float('inf')
        closest_ruma = None
        for ruma_id, ruma in self.rumas.items():
            if not ruma.is_active:
                continue
            dist = np.linalg.norm(np.array(centroid) - np.array(ruma.centroid))
            if dist < min_distance and dist < max_distance:
                min_distance = dist
                closest_ruma = ruma_id
        return closest_ruma, min_distance

    def add_candidate_ruma(self, mask, centroid, frame_count, frame_shape, transformer=None):
        key = f"candidate_{centroid[0]}_{centroid[1]}"
        if key not in self.candidate_rumas:
            self.candidate_rumas[key] = {
                'mask': mask,
                'centroid': centroid,
                'area': cv2.contourArea(mask.astype(np.int32)),
                'first_frame': frame_count,
                'confirmations': 1
            }
        else:
            self.candidate_rumas[key]['confirmations'] += 1

            if self.candidate_rumas[key]['confirmations'] >= 6:
                ruma_id = self.next_ruma_id
                area = cv2.contourArea(mask.astype(np.int32))
                new_ruma = RumaData(ruma_id, mask, area, centroid)

                # Calcular homografía si se provee el transformer
                if transformer:
                    ch, rh = transformer.transform_circle(centroid, new_ruma.radius)
                    new_ruma.centroid_homographic = tuple(map(float, ch))
                    new_ruma.radius_homographic = float(rh)

                self.rumas[ruma_id] = new_ruma
                self.store_ruma_summary(ruma_id, new_ruma, frame_shape)
                print(f"Nueva ruma creada: ID {ruma_id}")
                self.next_ruma_id += 1
                del self.candidate_rumas[key]

    def update_ruma(self, ruma_id, mask, frame_count):
        ruma = self.rumas[ruma_id]
        ruma.current_area = cv2.contourArea(mask.astype(np.int32))
        ruma.percentage = (ruma.current_area / ruma.initial_area) * 100
        ruma.last_seen_frame = frame_count
        ruma.centroid = (
            int(np.mean([p[0] for p in mask])),
            int(np.mean([p[1] for p in mask]))
        )
        if ruma.percentage <= 10:
            ruma.is_active = False
            print(f"Ruma {ruma_id} eliminada (porcentaje: {ruma.percentage:.1f}%)")

    def store_ruma_summary(self, ruma_id, ruma: RumaData, frame_shape):
        self.ruma_summary[ruma_id] = {
            'centroid': tuple(map(int, ruma.centroid)),
            'radius': round(ruma.radius, 2),
            'centroid_homographic': ruma.centroid_homographic,
            'radius_homographic': ruma.radius_homographic
        }
        self.new_ruma_created = (ruma_id, frame_shape)

    def clean_old_candidates(self, frame_count, max_age=100):
        to_remove = [
            key for key, candidate in self.candidate_rumas.items()
            if frame_count - candidate['first_frame'] > max_age
        ]
        for key in to_remove:
            del self.candidate_rumas[key]
