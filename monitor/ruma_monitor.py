import os
import cv2
import time
import json
import base64
import torch
import requests
import numpy as np
from ultralytics import YOLO

#--------- import de la clase ruma data -------#
from utils.geometry import is_point_in_polygon, calculate_intersection
from alerts.alert_manager import save_alert
from utils.draw import put_text_with_background, draw_zone_and_status
from monitor.ruma_tracker import RumaTracker
from monitor.object_tracker import ObjectTracker  # tracker object
from alerts.alert_info import RumaInfo # Dataclass para almacenar datos de rumas

#-----------#

class RumaMonitor:
    def __init__(self, model_det_path, model_seg_path, detection_zone, camera_sn, 
                 api_url, transformer, save_video=False, target_size=1024):
        """
        Inicializa el monitor de rumas

        Args:
            model_det_path: Ruta al modelo de detección (.pt o .engine)
            model_seg_path: Ruta al modelo de segmentación (.pt o .engine)
            detection_zone: Polígono que define la zona de detección
            camera_sn: Número de serie de la cámara
            api_url: URL de la API para alertas
            transformer: Transformador de homografía
            save_video: Si True, aplica dibujos visuales. Si False, solo procesa datos.
            target_size: Tamaño objetivo para resize (default: 1024)
        """
        self.api_url = api_url 
        self.model_det = YOLO(model_det_path)
        self.model_seg = YOLO(model_seg_path)
        self.detection_zone = detection_zone
        self.camera_sn = camera_sn
        self.enterprise = 'alma'
        self.save_video = save_video
        self.target_size = target_size  # NUEVO: tamaño objetivo
        
        # Variables para tracking de escala
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.original_size = None
        self.resized_size = None

        # Tracking de rumas
        self.tracker = RumaTracker()
        
        # Tracking de objetos (personas/vehículos)
        self.object_tracker = ObjectTracker(
            interaction_threshold=40,      # 40 frames = ~1.6s @ 25fps
            max_distance_match=100         # 100 píxeles máximo para matching
        )

        # Configuración de colores
        self.RUMA_COLOR = (0, 255, 0)
        self.PERSON_COLOR = (255, 255, 0)
        self.VEHICLE_COLOR = (0, 0, 255)
        self.TEXT_COLOR_WHITE = (255, 255, 255)
        self.TEXT_COLOR_GREEN = (0, 255, 0)
        self.TEXT_COLOR_RED = (0, 0, 255)

        self.transformer = transformer

        # Envio de datos
        self.send = True # Envio de datos a la nube
        self.save = False # Guardado de datos local

    def _resize_frame_if_needed(self, frame):
        """
        Redimensiona el frame si es mayor que target_size.
        Mantiene aspect ratio y calcula factores de escala.
        
        Returns:
            resized_frame, scale_x, scale_y
        """
        h, w = frame.shape[:2]
        
        # Si ya está dentro del límite, no hacer nada
        if max(h, w) <= self.target_size:
            return frame, 1.0, 1.0
        
        # Calcular nueva dimensión manteniendo aspect ratio
        if w > h:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))
        else:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))
        
        # Calcular factores de escala
        scale_x = w / new_w
        scale_y = h / new_h
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return resized, scale_x, scale_y

    def _scale_coordinates(self, coords, inverse=False):
        """
        Escala coordenadas según los factores de escala.
        
        Args:
            coords: Tupla (x, y) o lista de tuplas [(x1, y1), (x2, y2), ...]
            inverse: Si True, escala de pequeño a grande. Si False, de grande a pequeño.
        
        Returns:
            Coordenadas escaladas
        """
        if isinstance(coords, tuple):
            x, y = coords
            if inverse:
                return (int(x * self.scale_x), int(y * self.scale_y))
            else:
                return (int(x / self.scale_x), int(y / self.scale_y))
        elif isinstance(coords, list):
            return [self._scale_coordinates(c, inverse) for c in coords]
        elif isinstance(coords, np.ndarray):
            scaled = coords.copy()
            if inverse:
                scaled[:, 0] = coords[:, 0] * self.scale_x
                scaled[:, 1] = coords[:, 1] * self.scale_y
            else:
                scaled[:, 0] = coords[:, 0] / self.scale_x
                scaled[:, 1] = coords[:, 1] / self.scale_y
            return scaled.astype(np.int32)
        return coords

    def _scale_polygon(self, polygon):
        """Escala el polígono de detección según los factores de escala"""
        scaled = polygon.copy().astype(np.float32)
        scaled[:, 0] = polygon[:, 0] / self.scale_x
        scaled[:, 1] = polygon[:, 1] / self.scale_y
        return scaled.astype(np.int32)

    def process_detections(self, frame, frame_count):
        """
        Procesa las detecciones de personas y vehículos.
        Ahora trabaja con el frame redimensionado.
        """
        movement_alerts_to_send = set()
        objects_per_ruma = {}
        object_intersections = {}

        # Escalar zona de detección para el frame pequeño
        detection_zone_scaled = self._scale_polygon(self.detection_zone)

        result_det = self.model_det.track(frame, conf=0.5, persist=True, verbose=False)

        if (result_det is not None) and len(result_det) > 0:
            boxes = result_det[0].boxes

            if (boxes is not None) and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > 0.5:
                        yolo_track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            yolo_track_id = int(box.id[0])

                        # Centroide en frame pequeño
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        centroid = (center_x, center_y)
                        
                        object_type = 'person' if cls == 0 else 'vehicle'
                        
                        # Verificar si está en polígono (usando polígono escalado)
                        in_polygon = is_point_in_polygon(centroid, detection_zone_scaled)
                        
                        # Actualizar tracker (internamente usa coordenadas del frame pequeño)
                        internal_id = self.object_tracker.update_or_create_object(
                            yolo_track_id=yolo_track_id,
                            centroid=centroid,
                            bbox=(x1, y1, x2, y2),
                            object_type=object_type,
                            frame_count=frame_count,
                            in_polygon=in_polygon
                        )
                        
                        if self.object_tracker.check_movement_alert(internal_id):
                            movement_alerts_to_send.add(internal_id)
                        
                        # Dibujar solo si save_video está activo
                        if self.save_video:
                            color = self.PERSON_COLOR if cls == 0 else self.VEHICLE_COLOR
                            label = f'{object_type} ID:{internal_id}'
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            frame = put_text_with_background(
                                frame, label, (x1, y1 - 5),
                                color=self.TEXT_COLOR_WHITE, font_scale=0.6
                            )
                        
                        # Verificar interacción con rumas
                        for ruma_id, ruma in self.tracker.rumas.items():
                            if not ruma.is_active:
                                continue
                            # Las máscaras de rumas también están en escala pequeña
                            if calculate_intersection([x1, y1, x2, y2], ruma.initial_mask):
                                if internal_id not in object_intersections:
                                    object_intersections[internal_id] = set()
                                object_intersections[internal_id].add(ruma_id)
                                
                                if ruma_id not in objects_per_ruma:
                                    objects_per_ruma[ruma_id] = set()
                                objects_per_ruma[ruma_id].add(internal_id)

        return frame, movement_alerts_to_send, objects_per_ruma, object_intersections

    def process_segmentation(self, frame, frame_count, objects_per_ruma, object_intersections):
        """
        Procesa la segmentación de rumas.
        Trabaja con el frame redimensionado.
        """
        result_seg = self.model_seg(frame, conf=0.5, verbose=False)
        
        interaction_alerts_to_send = set()
        variation_alerts_to_send = set()
        max_frames_without_interaction = 15

        # Escalar zona de detección
        detection_zone_scaled = self._scale_polygon(self.detection_zone)

        if result_seg and len(result_seg) > 0:
            for r in result_seg:
                if r.masks is not None:
                    masks = r.masks.xy

                    for mask in masks:
                        centroid_x = int(np.mean([p[0] for p in mask]))
                        centroid_y = int(np.mean([p[1] for p in mask]))
                        centroid = (centroid_x, centroid_y)

                        # Verificar si está en zona (usando polígono escalado)
                        if not is_point_in_polygon(centroid, detection_zone_scaled):
                            continue

                        closest_ruma_id, distance = self.tracker.find_closest_ruma(centroid)

                        if closest_ruma_id is not None:
                            self.tracker.update_ruma(closest_ruma_id, mask, frame_count)
                            ruma = self.tracker.rumas[closest_ruma_id]

                            # Dibujar solo si save_video está activo
                            if self.save_video:
                                overlay = frame.copy()
                                cv2.fillPoly(overlay, [mask.astype(np.int32)], self.RUMA_COLOR)
                                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                            is_interacting = closest_ruma_id in objects_per_ruma and len(objects_per_ruma[closest_ruma_id]) > 0
                            
                            if ruma.was_interacting and not is_interacting:
                                if ruma.should_send_variation_alert():
                                    variation_alerts_to_send.add(closest_ruma_id)
                            
                            ruma.was_interacting = is_interacting

                            if is_interacting:
                                ruma.frames_without_interaction = 0
                                ruma.last_stable_percentage = ruma.percentage
                                
                                for internal_id in objects_per_ruma[closest_ruma_id]:
                                    intersecting_rumas = object_intersections.get(internal_id, set())
                                    should_alert, confirmed_ruma = self.object_tracker.update_interaction(
                                        internal_id, intersecting_rumas
                                    )
                                    if should_alert:
                                        interaction_alerts_to_send.add((internal_id, confirmed_ruma))
                            else:
                                ruma.frames_without_interaction += 1
                                if ruma.frames_without_interaction < max_frames_without_interaction:
                                    ruma.last_stable_percentage = max(ruma.last_stable_percentage, ruma.percentage)

                            ruma.percentage = min(ruma.percentage, 100)
                            ruma.last_stable_percentage = min(ruma.last_stable_percentage, 100)

                            if ruma.frames_without_interaction >= max_frames_without_interaction:
                                display_percentage = ruma.last_stable_percentage
                                draw_ruma_variation = False
                            else:
                                display_percentage = ruma.percentage
                                draw_ruma_variation = True

                            if self.save_video:
                                label_text = f"R{ruma.id} | {display_percentage:.1f}%"
                                frame = put_text_with_background(
                                    frame, label_text, ruma.label_position,
                                    font_scale=0.6, color=self.TEXT_COLOR_WHITE
                                )

                        else:
                            # Nueva ruma candidata
                            self.tracker.add_candidate_ruma(mask, centroid, frame_count, frame.shape, self.transformer)

        self.tracker.clean_old_candidates(frame_count)

        return frame, interaction_alerts_to_send, variation_alerts_to_send

    def process_frame(self, frame, frame_count, fps):
        """
        Procesa un frame completo.
        Ahora hace resize si es necesario.
        """
        if frame is None or frame.size == 0:
            print(f"[WARN] Frame {frame_count} inválido o vacío, saltando...")
            if hasattr(self, '_last_valid_frame_shape'):
                return np.zeros(self._last_valid_frame_shape, dtype=np.uint8)
            else:
                return np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Guardar frame original para alertas
        original_frame = frame.copy()
        self._last_valid_frame_shape = frame.shape
        
        # Redimensionar si es necesario
        resized_frame, self.scale_x, self.scale_y = self._resize_frame_if_needed(frame)
        
        # Si es el primer frame, ajustar zona de detección una vez
        if self.original_size is None:
            self.original_size = frame.shape[:2]
            self.resized_size = resized_frame.shape[:2]
            print(f"[INFO] Frame original: {self.original_size}, Frame procesado: {self.resized_size}")
            print(f"[INFO] Factores de escala: x={self.scale_x:.2f}, y={self.scale_y:.2f}")
        
        # Procesar con frame redimensionado
        frame_with_drawings = resized_frame.copy()

        # Procesar detecciones y segmentación
        frame_with_drawings, movement_alerts, objects_per_ruma, object_intersections = self.process_detections(
            frame_with_drawings, frame_count
        )

        frame_with_drawings, interaction_alerts, variation_alerts = self.process_segmentation(
            frame_with_drawings, frame_count, objects_per_ruma, object_intersections
        )

        # Dibujar zona y estado solo si save_video está activo
        if self.save_video:
            detection_zone_scaled = self._scale_polygon(self.detection_zone)
            has_movement = len(movement_alerts) > 0
            has_interaction = len(interaction_alerts) > 0
            has_variation = len(variation_alerts) > 0
            
            frame_with_drawings = draw_zone_and_status(
                frame_with_drawings,
                detection_zone_scaled,
                has_movement,
                has_interaction,
                has_variation,
                TEXT_COLOR_RED=self.TEXT_COLOR_RED,
                TEXT_COLOR_GREEN=self.TEXT_COLOR_GREEN
            )

        # === ENVIAR ALERTAS (usar frame original para las imágenes) ===
        
        # 1. Alertas de movimiento
        for internal_id in movement_alerts:
            ruma_data = RumaInfo(
                id=None, percent=None, centroid=None, radius=None,
                centroid_homographic=None, radius_homographic=None
            )
            
            save_alert(
                alert_type='movimiento_zona',
                ruma_data=ruma_data,
                frame=original_frame,  # Usar frame original
                frame_count=frame_count,
                fps=fps,
                camera_sn=self.camera_sn,
                enterprise=self.enterprise,
                api_url=self.api_url,
                send=self.send,
                save=self.save,
                ruma_summary=self.tracker.ruma_summary,
                frame_shape=original_frame.shape,
                detection_zone=self.detection_zone  # Usar polígono original
            )
        
        # 2. Alertas de interacción
        for (internal_id, ruma_id) in interaction_alerts:
            if ruma_id in self.tracker.rumas:
                ruma = self.tracker.rumas[ruma_id]
                
                # Escalar coordenadas de ruma a tamaño original
                centroid_original = self._scale_coordinates(ruma.centroid, inverse=True)
                radius_original = ruma.radius * max(self.scale_x, self.scale_y)
                
                ruma_data = RumaInfo(
                    id=ruma.id,
                    percent=ruma.last_stable_percentage,
                    centroid=centroid_original,
                    radius=radius_original,
                    centroid_homographic=ruma.centroid_homographic,
                    radius_homographic=ruma.radius_homographic
                )
                
                save_alert(
                    alert_type='interaccion_rumas',
                    ruma_data=ruma_data,
                    frame=original_frame,
                    frame_count=frame_count,
                    fps=fps,
                    camera_sn=self.camera_sn,
                    enterprise=self.enterprise,
                    api_url=self.api_url,
                    send=self.send,
                    save=self.save,
                    ruma_summary=self.tracker.ruma_summary,
                    frame_shape=original_frame.shape,
                    detection_zone=self.detection_zone
                )
        
        # 3. Alertas de variación
        for ruma_id in variation_alerts:
            if ruma_id in self.tracker.rumas:
                ruma = self.tracker.rumas[ruma_id]
                
                centroid_original = self._scale_coordinates(ruma.centroid, inverse=True)
                radius_original = ruma.radius * max(self.scale_x, self.scale_y)
                
                ruma_data = RumaInfo(
                    id=ruma.id,
                    percent=ruma.percentage,
                    centroid=centroid_original,
                    radius=radius_original,
                    centroid_homographic=ruma.centroid_homographic,
                    radius_homographic=ruma.radius_homographic
                )
                
                save_alert(
                    alert_type='variacion_rumas',
                    ruma_data=ruma_data,
                    frame=original_frame,
                    frame_count=frame_count,
                    fps=fps,
                    camera_sn=self.camera_sn,
                    enterprise=self.enterprise,
                    api_url=self.api_url,
                    send=self.send,
                    save=self.save,
                    ruma_summary=self.tracker.ruma_summary,
                    frame_shape=original_frame.shape,
                    detection_zone=self.detection_zone
                )
        
        # 4. Nueva ruma
        if self.tracker.new_ruma_created:
            ruma_id, frame_shape = self.tracker.new_ruma_created
            ruma = self.tracker.rumas[ruma_id]
            
            centroid_original = self._scale_coordinates(ruma.centroid, inverse=True)
            radius_original = ruma.radius * max(self.scale_x, self.scale_y)
            
            ruma_data = RumaInfo(
                id=ruma.id,
                percent=100.0,
                centroid=centroid_original,
                radius=radius_original,
                centroid_homographic=ruma.centroid_homographic,
                radius_homographic=ruma.radius_homographic
            )

            save_alert(
                alert_type='nueva_ruma',
                ruma_data=ruma_data,
                frame=original_frame,
                frame_count=frame_count,
                fps=fps,
                camera_sn=self.camera_sn,
                enterprise=self.enterprise,
                api_url=self.api_url,
                send=self.send,
                save=self.save,
                ruma_summary=self.tracker.ruma_summary,
                frame_shape=original_frame.shape,
                detection_zone=self.detection_zone
            )

            self.tracker.new_ruma_created = None
        
        self.object_tracker.cleanup_old_objects(frame_count)

        # Si save_video está activo, redimensionar de vuelta al tamaño original
        if self.save_video and (self.scale_x != 1.0 or self.scale_y != 1.0):
            frame_with_drawings = cv2.resize(
                frame_with_drawings, 
                (original_frame.shape[1], original_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return frame_with_drawings if self.save_video else original_frame