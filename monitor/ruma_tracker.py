import cv2
import numpy as np
from monitor.ruma_data import RumaData  # Asegúrate que este archivo contenga el dataclass

class RumaTracker:
    def __init__(self):
        self.rumas = {}  # ruma_id: RumaData
        self.candidate_rumas = {}  # clave: str → datos temporales
        self.next_ruma_id = 1
        self.ruma_summary = {}
        self.new_ruma_ids = []  # IDs de rumas creadas recientemente

    def find_closest_ruma(self, centroid, max_distance=50):
        """Encuentra la ruma activa más cercana a un centroide dado"""
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

    def add_candidate_ruma(self, mask, centroid, frame_count, frame_shape):
        """Agrega o confirma una nueva ruma candidata"""
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
                #radius = self._calculate_radius(mask, centroid)
                #coords = mask.astype(int).tolist()  # asegúrate que sea una lista de puntos

                new_ruma = RumaData(
                    id=ruma_id,
                    centroid=centroid
                )

                self.rumas[ruma_id] = new_ruma

                self.store_ruma_summary(ruma_id, mask, centroid, frame_shape)
                self.new_ruma_ids.append(ruma_id)

                print(f"✅ Nueva ruma creada: ID {ruma_id}")
                self.next_ruma_id += 1
                del self.candidate_rumas[key]

    def update_ruma(self, ruma_id, mask, frame_count):
        """Actualiza una ruma existente con nueva máscara"""
        ruma = self.rumas[ruma_id]
        ruma.current_area = cv2.contourArea(mask.astype(np.int32))
        ruma.percentage = (ruma.current_area / ruma.initial_area) * 100
        ruma.last_seen_frame = frame_count

        # Actualizar centroide
        ruma.centroid = (
            int(np.mean([p[0] for p in mask])),
            int(np.mean([p[1] for p in mask]))
        )

        if ruma.percentage <= 10:
            ruma.is_active = False
            print(f"⚠️ Ruma {ruma_id} eliminada (porcentaje: {ruma.percentage:.1f}%)")

    def store_ruma_summary(self, ruma_id, mask, centroid, frame_shape):
        """Guarda metadatos de la ruma (para alertas, etc.)"""
        distances = [np.linalg.norm(np.array(centroid) - np.array(p)) for p in mask]
        radius = float(np.mean(distances))

        self.ruma_summary[ruma_id] = {
            'centroid': tuple(map(int, centroid)),
            'radius': round(radius, 2)
        }

    def get_and_clear_new_rumas(self):
        """Devuelve y limpia los IDs de rumas recién creadas"""
        new_ids = self.new_ruma_ids.copy()
        self.new_ruma_ids.clear()
        return new_ids

    def clean_old_candidates(self, frame_count, max_age=100):
        """Elimina rumas candidatas que no se confirmaron a tiempo"""
        to_remove = []
        for key, candidate in self.candidate_rumas.items():
            if frame_count - candidate['first_frame'] > max_age:
                to_remove.append(key)

        for key in to_remove:
            del self.candidate_rumas[key]
    def _calculate_radius(self, mask, centroid):
        distances = [np.linalg.norm(np.array(centroid) - np.array(p)) for p in mask]
        return float(np.mean(distances)) if distances else 0.0
