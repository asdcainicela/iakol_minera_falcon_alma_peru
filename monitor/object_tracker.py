import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Set, Deque
from collections import deque

@dataclass
class TrackedObject:
    """Representa un objeto trackeado (persona o vehículo)"""
    internal_id: int
    yolo_track_id: Optional[int]
    object_type: str  # 'person' | 'vehicle'
    centroid: Tuple[int, int]
    bbox_area: float
    last_seen_frame: int
    centroid_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))
    
    # Estados de alerta
    in_zone: bool = False
    alert_sent_movement: bool = False
    current_ruma_id: Optional[int] = None
    frames_in_current_ruma: int = 0
    interacted_rumas: Set[int] = field(default_factory=set)
    
    # NUEVO: Tracking de velocidad
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) en px/frame
    frames_missing: int = 0  # Frames consecutivos sin detección
    
    def __post_init__(self):
        """Inicializar historial con posición actual"""
        if not self.centroid_history:
            self.centroid_history.append(self.centroid)
    
    def calculate_velocity(self):
        """
        Calcula la velocidad promedio basada en el historial.
        Usa los últimos 3 puntos para suavizar.
        """
        if len(self.centroid_history) < 2:
            self.velocity = (0.0, 0.0)
            return
        
        # Tomar últimos 3 puntos (o menos si no hay suficientes)
        points = list(self.centroid_history)[-3:]
        
        # Calcular velocidades entre puntos consecutivos
        velocities = []
        for i in range(len(points) - 1):
            vx = points[i+1][0] - points[i][0]
            vy = points[i+1][1] - points[i][1]
            velocities.append((vx, vy))
        
        # Promediar velocidades
        if velocities:
            avg_vx = sum(v[0] for v in velocities) / len(velocities)
            avg_vy = sum(v[1] for v in velocities) / len(velocities)
            self.velocity = (avg_vx, avg_vy)
    
    def predict_position(self, frames_ahead: int = 1) -> Tuple[int, int]:
        """
        Predice la posición futura basada en velocidad.
        
        Args:
            frames_ahead: Cuántos frames hacia adelante predecir
        
        Returns:
            Posición predicha (x, y)
        """
        pred_x = int(self.centroid[0] + self.velocity[0] * frames_ahead)
        pred_y = int(self.centroid[1] + self.velocity[1] * frames_ahead)
        return (pred_x, pred_y)


class ObjectTracker:
    """Trackea personas y vehículos con predicción de movimiento"""
    
    def __init__(self, interaction_threshold=40, max_distance_match=150):
        """
        Args:
            interaction_threshold: Frames consecutivos para confirmar interacción
            max_distance_match: Distancia máxima (px) para considerar mismo objeto
        """
        self.interaction_threshold = interaction_threshold
        self.max_distance_match = max_distance_match
        self.next_internal_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        
        # ⚙️ CONFIGURACIÓN MEJORADA
        self.max_frames_missing = 15          # 15 frames @ 2fps = 7.5s de gracia
        self.prediction_frames = 3            # Predecir hasta 3 frames adelante
        self.bbox_size_tolerance = 0.4        # ±40% tolerancia en tamaño
        self.use_velocity_matching = True     # Usar predicción de velocidad
        
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _is_size_similar(self, area1: float, area2: float) -> bool:
        """Verifica si dos áreas son similares dentro de la tolerancia"""
        if area1 == 0 or area2 == 0:
            return False
        ratio = min(area1, area2) / max(area1, area2)
        return ratio >= (1 - self.bbox_size_tolerance)
    
    def _calculate_direction_similarity(self, obj: TrackedObject, new_centroid: Tuple[int, int]) -> float:
        """
        Calcula qué tan bien la nueva posición coincide con la dirección de movimiento.
        
        Returns:
            Score de 0 a 1 (1 = perfecta alineación con velocidad)
        """
        if abs(obj.velocity[0]) < 1 and abs(obj.velocity[1]) < 1:
            return 0.5  # Sin movimiento significativo, neutro
        
        # Vector desde última posición a nueva posición
        actual_dx = new_centroid[0] - obj.centroid[0]
        actual_dy = new_centroid[1] - obj.centroid[1]
        
        # Producto punto normalizado (cosine similarity)
        vel_mag = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
        actual_mag = np.sqrt(actual_dx**2 + actual_dy**2)
        
        if vel_mag < 0.1 or actual_mag < 0.1:
            return 0.5
        
        dot_product = (obj.velocity[0] * actual_dx + obj.velocity[1] * actual_dy)
        similarity = dot_product / (vel_mag * actual_mag)
        
        # Convertir de [-1, 1] a [0, 1]
        return (similarity + 1) / 2
    
    def find_matching_object(self, 
                            centroid: Tuple[int, int], 
                            bbox_area: float, 
                            object_type: str,
                            yolo_track_id: Optional[int]) -> Optional[int]:
        """
        Busca un objeto existente con PREDICCIÓN DE MOVIMIENTO.
        
        Estrategia mejorada:
        1. Buscar por yolo_track_id si existe
        2. Buscar por posición actual
        3. Buscar por posición PREDICHA (para objetos perdidos temporalmente)
        """
        # === ESTRATEGIA 1: Buscar por YOLO track_id ===
        if yolo_track_id is not None:
            for internal_id, obj in self.tracked_objects.items():
                if obj.yolo_track_id == yolo_track_id and obj.object_type == object_type:
                    return internal_id
        
        # === ESTRATEGIA 2: Buscar por proximidad + tipo + tamaño ===
        candidates = []
        
        for internal_id, obj in self.tracked_objects.items():
            # Mismo tipo de objeto
            if obj.object_type != object_type:
                continue
            
            # Calcular distancia a posición ACTUAL
            distance_actual = self._calculate_distance(centroid, obj.centroid)
            
            # Si el objeto fue visto recientemente, usar distancia normal
            if obj.frames_missing <= 3:
                if distance_actual <= self.max_distance_match and self._is_size_similar(bbox_area, obj.bbox_area):
                    # Calcular score de dirección
                    direction_score = self._calculate_direction_similarity(obj, centroid)
                    # Score combinado (distancia baja = score alto)
                    score = (1 - distance_actual / self.max_distance_match) * 0.7 + direction_score * 0.3
                    candidates.append((internal_id, score, distance_actual))
            
            # === ESTRATEGIA 3: Predicción para objetos perdidos ===
            elif obj.frames_missing <= self.max_frames_missing and self.use_velocity_matching:
                # Predecir posición basada en frames perdidos
                predicted_pos = obj.predict_position(frames_ahead=obj.frames_missing)
                distance_predicted = self._calculate_distance(centroid, predicted_pos)
                
                # Usar distancia más flexible para predicciones
                max_dist_predicted = self.max_distance_match * 1.5
                
                if distance_predicted <= max_dist_predicted:
                    # Score basado en qué tan bien coincide con predicción
                    score = (1 - distance_predicted / max_dist_predicted) * 0.5  # Score reducido para predicciones
                    candidates.append((internal_id, score, distance_predicted))
        
        # Retornar el mejor candidato
        if candidates:
            best_match = max(candidates, key=lambda x: x[1])  # Máximo score
            return best_match[0]
        
        return None
    
    def update_or_create_object(self,
                                yolo_track_id: Optional[int],
                                centroid: Tuple[int, int],
                                bbox: Tuple[int, int, int, int],
                                object_type: str,
                                frame_count: int,
                                in_polygon: bool) -> int:
        """
        Actualiza un objeto existente o crea uno nuevo.
        Con predicción de velocidad mejorada.
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Buscar matching con predicción
        internal_id = self.find_matching_object(centroid, bbox_area, object_type, yolo_track_id)
        
        if internal_id is not None:
            # ✅ ACTUALIZAR objeto existente
            obj = self.tracked_objects[internal_id]
            
            # Actualizar datos
            obj.yolo_track_id = yolo_track_id
            obj.centroid = centroid
            obj.bbox_area = bbox_area
            obj.last_seen_frame = frame_count
            obj.centroid_history.append(centroid)
            obj.in_zone = in_polygon
            obj.frames_missing = 0  # ← RESETEAR contador de frames perdidos
            
            # Recalcular velocidad con nueva posición
            obj.calculate_velocity()
            
        else:
            # 🆕 CREAR nuevo objeto
            internal_id = self.next_internal_id
            self.next_internal_id += 1
            
            new_obj = TrackedObject(
                internal_id=internal_id,
                yolo_track_id=yolo_track_id,
                object_type=object_type,
                centroid=centroid,
                bbox_area=bbox_area,
                last_seen_frame=frame_count,
                in_zone=in_polygon
            )
            
            self.tracked_objects[internal_id] = new_obj
            print(f"[ObjectTracker] Nuevo objeto creado: ID={internal_id}, tipo={object_type}")
        
        return internal_id
    
    def check_movement_alert(self, internal_id: int) -> bool:
        """Verifica si debe enviar alerta de movimiento en zona"""
        if internal_id not in self.tracked_objects:
            return False
        
        obj = self.tracked_objects[internal_id]
        
        if obj.in_zone and not obj.alert_sent_movement:
            obj.alert_sent_movement = True
            return True
        
        if not obj.in_zone:
            obj.alert_sent_movement = False
        
        return False
    
    def update_interaction(self, 
                          internal_id: int, 
                          intersecting_ruma_ids: Set[int]) -> Tuple[bool, Optional[int]]:
        """Actualiza el estado de interacción con rumas"""
        if internal_id not in self.tracked_objects:
            return False, None
        
        obj = self.tracked_objects[internal_id]
        
        if not intersecting_ruma_ids:
            obj.current_ruma_id = None
            obj.frames_in_current_ruma = 0
            return False, None
        
        current_ruma = next(iter(intersecting_ruma_ids))
        
        if obj.current_ruma_id != current_ruma:
            obj.current_ruma_id = current_ruma
            obj.frames_in_current_ruma = 1
            return False, None
        
        obj.frames_in_current_ruma += 1
        
        if (obj.frames_in_current_ruma >= self.interaction_threshold and 
            current_ruma not in obj.interacted_rumas):
            obj.interacted_rumas.add(current_ruma)
            print(f"[ObjectTracker] Interacción confirmada: objeto {internal_id} con ruma {current_ruma}")
            return True, current_ruma
        
        return False, None
    
    def cleanup_old_objects(self, current_frame: int):
        """
        Elimina objetos con GRACIA EXTENDIDA para objetos perdidos temporalmente.
        """
        to_remove = []
        
        for internal_id, obj in self.tracked_objects.items():
            frames_missing = current_frame - obj.last_seen_frame
            
            # Incrementar contador de frames perdidos
            obj.frames_missing = frames_missing
            
            # Eliminar solo si excede el límite de gracia
            if frames_missing > self.max_frames_missing:
                to_remove.append(internal_id)
        
        for internal_id in to_remove:
            del self.tracked_objects[internal_id]
        
        if to_remove:
            print(f"[ObjectTracker] Limpieza: {len(to_remove)} objetos eliminados")
    
    def get_active_objects(self) -> Dict[int, TrackedObject]:
        """Retorna diccionario de objetos activos"""
        return self.tracked_objects.copy()