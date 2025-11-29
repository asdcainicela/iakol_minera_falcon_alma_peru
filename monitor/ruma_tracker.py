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
        
        # ⚙️ PARÁMETROS DE VALIDACIÓN (ajustados para 2 FPS)
        self.min_confirmations = 3           # Mínimo 3 detecciones
        self.min_frames_stable = 15          # 15 frames @ 2 FPS = 7.5s
        self.max_movement_px = 50            # Máximo 50px de movimiento
        self.max_candidate_age = 100         # 100 frames @ 2 FPS = 50s
        self.frames_since_seen_limit = 50    # No visto en 50 frames = 25s

    def _calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def find_closest_ruma(self, centroid, max_distance=50):
        min_distance = float('inf')
        closest_ruma = None
        for ruma_id, ruma in self.rumas.items():
            if not ruma.is_active:
                continue
            dist = self._calculate_distance(centroid, ruma.centroid)
            if dist < min_distance and dist < max_distance:
                min_distance = dist
                closest_ruma = ruma_id
        return closest_ruma, min_distance

    def add_candidate_ruma(self, mask, centroid, frame_count, frame_shape, transformer=None):
        """
        Agrega o actualiza una ruma candidata.
        
        CRITERIOS DE VALIDACIÓN:
        1. Debe permanecer estable (< max_movement_px) durante min_frames_stable frames
        2. Si se mueve demasiado, se reinicia el contador
        """
        key = f"candidate_{centroid[0]}_{centroid[1]}"
        
        # Buscar candidato cercano existente
        existing_key = None
        min_dist = float('inf')
        
        for existing_key_iter, candidate in self.candidate_rumas.items():
            dist = self._calculate_distance(centroid, candidate['centroid'])
            if dist < min_dist and dist < 50:  # 50px = radio de búsqueda
                min_dist = dist
                existing_key = existing_key_iter
        
        if existing_key is not None:
            # ✅ CANDIDATO EXISTENTE ENCONTRADO
            candidate = self.candidate_rumas[existing_key]
            
            # Calcular movimiento desde la posición inicial
            movement = self._calculate_distance(centroid, candidate['initial_centroid'])
            
            if movement <= self.max_movement_px:
                # ✅ SE MOVIÓ POCO - Incrementar confirmaciones
                candidate['confirmations'] += 1
                candidate['last_seen_frame'] = frame_count
                candidate['centroid'] = centroid  # Actualizar centroide actual
                candidate['mask'] = mask  # Actualizar máscara
                
                # Calcular frames transcurridos
                frames_elapsed = frame_count - candidate['first_frame']
                
                # ✅ VALIDACIÓN: Estable por suficiente tiempo
                if (candidate['confirmations'] >= self.min_confirmations and
                    frames_elapsed >= self.min_frames_stable):
                    
                    # 🎉 CONFIRMAR COMO RUMA REAL
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
                    
                    print(f"✅ Nueva ruma creada: ID {ruma_id} "
                          f"(estable por {frames_elapsed} frames, "
                          f"movimiento total: {movement:.1f}px)")
                    
                    self.next_ruma_id += 1
                    del self.candidate_rumas[existing_key]
            else:
                # ❌ SE MOVIÓ DEMASIADO - Reiniciar validación
                print(f"⚠️ Candidato {existing_key} se movió {movement:.1f}px - Reiniciando")
                candidate['initial_centroid'] = centroid
                candidate['centroid'] = centroid
                candidate['first_frame'] = frame_count
                candidate['confirmations'] = 1
                candidate['last_seen_frame'] = frame_count
        else:
            # 🆕 NUEVO CANDIDATO
            self.candidate_rumas[key] = {
                'mask': mask,
                'centroid': centroid,
                'initial_centroid': centroid,  # ← Guardar posición inicial
                'area': cv2.contourArea(mask.astype(np.int32)),
                'first_frame': frame_count,
                'last_seen_frame': frame_count,
                'confirmations': 1
            }
            print(f"🆕 Nuevo candidato en ({centroid[0]}, {centroid[1]})")

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
            print(f"❌ Ruma {ruma_id} eliminada (porcentaje: {ruma.percentage:.1f}%)")

    def store_ruma_summary(self, ruma_id, ruma: RumaData, frame_shape):
        self.ruma_summary[ruma_id] = {
            'centroid': tuple(map(int, ruma.centroid)),
            'radius': round(ruma.radius, 2),
            'centroid_homographic': ruma.centroid_homographic,
            'radius_homographic': ruma.radius_homographic
        }
        self.new_ruma_created = (ruma_id, frame_shape)

    def clean_old_candidates(self, frame_count):
        """Elimina candidatos muy antiguos o que no han sido vistos recientemente"""
        to_remove = []
        
        for key, candidate in self.candidate_rumas.items():
            frames_since_seen = frame_count - candidate['last_seen_frame']
            total_age = frame_count - candidate['first_frame']
            
            # Eliminar si no se ha visto recientemente O es muy antiguo
            if frames_since_seen > self.frames_since_seen_limit or total_age > self.max_candidate_age:
                print(f"🧹 Eliminando candidato: {key} "
                      f"(edad: {total_age} frames, "
                      f"no visto hace: {frames_since_seen} frames)")
                to_remove.append(key)
        
        for key in to_remove:
            del self.candidate_rumas[key]