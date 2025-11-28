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
    
    def __post_init__(self):
        """Inicializar historial con posición actual"""
        if not self.centroid_history:
            self.centroid_history.append(self.centroid)


class ObjectTracker:
    """Trackea personas y vehículos de forma robusta ante movimiento de cámara PTZ"""
    
    def __init__(self, interaction_threshold=40, max_distance_match=100):
        """
        Args:
            interaction_threshold: Frames consecutivos para confirmar interacción con ruma
            max_distance_match: Distancia máxima (px) para considerar mismo objeto
        """
        self.interaction_threshold = interaction_threshold
        self.max_distance_match = max_distance_match
        self.next_internal_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        
        # Configuración
        self.max_frames_missing = 100  # Eliminar objetos no vistos en X frames
        self.bbox_size_tolerance = 0.3  # ±30% tolerancia en tamaño
        
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _is_size_similar(self, area1: float, area2: float) -> bool:
        """Verifica si dos áreas son similares dentro de la tolerancia"""
        if area1 == 0 or area2 == 0:
            return False
        ratio = min(area1, area2) / max(area1, area2)
        return ratio >= (1 - self.bbox_size_tolerance)
    
    def find_matching_object(self, 
                            centroid: Tuple[int, int], 
                            bbox_area: float, 
                            object_type: str,
                            yolo_track_id: Optional[int]) -> Optional[int]:
        """
        Busca un objeto existente que coincida con la detección actual.
        
        Estrategia:
        1. Buscar por yolo_track_id si existe
        2. Si no, buscar por proximidad + tipo + tamaño similar
        
        Returns:
            internal_id del objeto encontrado o None
        """
        # Estrategia 1: Buscar por YOLO track_id
        if yolo_track_id is not None:
            for internal_id, obj in self.tracked_objects.items():
                if obj.yolo_track_id == yolo_track_id and obj.object_type == object_type:
                    return internal_id
        
        # Estrategia 2: Buscar por proximidad + tipo + tamaño
        best_match = None
        min_distance = float('inf')
        
        for internal_id, obj in self.tracked_objects.items():
            # Mismo tipo de objeto
            if obj.object_type != object_type:
                continue
            
            # Calcular distancia
            distance = self._calculate_distance(centroid, obj.centroid)
            
            # Verificar si está dentro del umbral
            if distance > self.max_distance_match:
                continue
            
            # Verificar tamaño similar
            if not self._is_size_similar(bbox_area, obj.bbox_area):
                continue
            
            # Guardar el más cercano
            if distance < min_distance:
                min_distance = distance
                best_match = internal_id
        
        return best_match
    
    def update_or_create_object(self,
                                yolo_track_id: Optional[int],
                                centroid: Tuple[int, int],
                                bbox: Tuple[int, int, int, int],
                                object_type: str,
                                frame_count: int,
                                in_polygon: bool) -> int:
        """
        Actualiza un objeto existente o crea uno nuevo.
        
        Args:
            yolo_track_id: ID del tracker de YOLO (puede ser None)
            centroid: Centro del bbox (x, y)
            bbox: Bounding box (x1, y1, x2, y2)
            object_type: 'person' o 'vehicle'
            frame_count: Frame actual
            in_polygon: Si está dentro del polígono de detección
            
        Returns:
            internal_id del objeto
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Buscar matching
        internal_id = self.find_matching_object(centroid, bbox_area, object_type, yolo_track_id)
        
        if internal_id is not None:
            # Actualizar objeto existente
            obj = self.tracked_objects[internal_id]
            obj.yolo_track_id = yolo_track_id  # Actualizar con el nuevo track_id de YOLO
            obj.centroid = centroid
            obj.bbox_area = bbox_area
            obj.last_seen_frame = frame_count
            obj.centroid_history.append(centroid)
            obj.in_zone = in_polygon
        else:
            # Crear nuevo objeto
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
        """
        Verifica si debe enviar alerta de movimiento en zona.
        
        Solo envía alerta la PRIMERA VEZ que el objeto entra al polígono.
        
        Returns:
            True si debe enviar alerta
        """
        if internal_id not in self.tracked_objects:
            return False
        
        obj = self.tracked_objects[internal_id]
        
        # Si está en zona y no se ha enviado alerta
        if obj.in_zone and not obj.alert_sent_movement:
            obj.alert_sent_movement = True
            return True
        
        # Si sale de la zona, resetear para permitir nueva alerta si vuelve a entrar
        if not obj.in_zone:
            obj.alert_sent_movement = False
        
        return False
    
    def update_interaction(self, 
                          internal_id: int, 
                          intersecting_ruma_ids: Set[int]) -> Tuple[bool, Optional[int]]:
        """
        Actualiza el estado de interacción con rumas.
        
        Args:
            internal_id: ID del objeto
            intersecting_ruma_ids: Set de IDs de rumas con las que intersecta
            
        Returns:
            (should_alert, ruma_id): True si debe enviar alerta, y el ID de la ruma
        """
        if internal_id not in self.tracked_objects:
            return False, None
        
        obj = self.tracked_objects[internal_id]
        
        # Si no está tocando ninguna ruma
        if not intersecting_ruma_ids:
            obj.current_ruma_id = None
            obj.frames_in_current_ruma = 0
            return False, None
        
        # Tomar la primera ruma con la que intersecta
        current_ruma = next(iter(intersecting_ruma_ids))
        
        # Si cambió de ruma, resetear contador
        if obj.current_ruma_id != current_ruma:
            obj.current_ruma_id = current_ruma
            obj.frames_in_current_ruma = 1
            return False, None
        
        # Incrementar contador de frames en la misma ruma
        obj.frames_in_current_ruma += 1
        
        # Verificar si alcanzó el umbral y no se ha alertado antes
        if (obj.frames_in_current_ruma >= self.interaction_threshold and 
            current_ruma not in obj.interacted_rumas):
            obj.interacted_rumas.add(current_ruma)
            print(f"[ObjectTracker] Interacción confirmada: objeto {internal_id} con ruma {current_ruma} ({obj.frames_in_current_ruma} frames)")
            return True, current_ruma
        
        return False, None
    
    def cleanup_old_objects(self, current_frame: int):
        """
        Elimina objetos que no se han visto en mucho tiempo.
        
        Args:
            current_frame: Frame actual
        """
        to_remove = []
        
        for internal_id, obj in self.tracked_objects.items():
            frames_missing = current_frame - obj.last_seen_frame
            if frames_missing > self.max_frames_missing:
                to_remove.append(internal_id)
        
        for internal_id in to_remove:
            del self.tracked_objects[internal_id]
            if to_remove:
                print(f"[ObjectTracker] Limpieza: {len(to_remove)} objetos eliminados")
    
    def get_active_objects(self) -> Dict[int, TrackedObject]:
        """Retorna diccionario de objetos activos"""
        return self.tracked_objects.copy()