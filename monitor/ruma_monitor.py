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
                 api_url, transformer, save_video=False):
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
        """
        self.api_url = api_url 
        
        # Cargar modelos (YOLO detecta automáticamente el formato)
        print(f"[INFO] Cargando modelo de detección: {model_det_path}")
        self.model_det = YOLO(model_det_path, task='detect')
        
        print(f"[INFO] Cargando modelo de segmentación: {model_seg_path}")
        self.model_seg = YOLO(model_seg_path, task='segment')
        
        self.detection_zone = detection_zone
        self.camera_sn = camera_sn
        self.enterprise = 'alma'
        self.save_video = save_video

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
        
        print("[INFO] RumaMonitor inicializado correctamente")

    def process_detections(self, frame, frame_count):
        """
        Procesa las detecciones de personas y vehículos.
        
        Returns:
            - frame: Frame con dibujos (si save_video=True)
            - movement_alerts_to_send: Set de internal_ids que deben generar alerta de movimiento
            - objects_per_ruma: Dict[ruma_id -> Set[internal_ids]] de objetos que intersectan cada ruma
        """
        movement_alerts_to_send = set()
        objects_per_ruma = {}  # ruma_id -> set(internal_ids)
        
        # Inicializar diccionario para tracking de intersecciones
        object_intersections = {}  # internal_id -> set(ruma_ids)

        result_det = self.model_det.track(frame, conf=0.5, persist=True, verbose=False)

        if (result_det is not None) and len(result_det) > 0:
            boxes = result_det[0].boxes

            if (boxes is not None) and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > 0.5:
                        # Extraer track_id de YOLO (puede ser None)
                        yolo_track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            yolo_track_id = int(box.id[0])

                        # Calcular centroide y área del bbox
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        centroid = (center_x, center_y)
                        
                        # Tipo de objeto
                        object_type = 'person' if cls == 0 else 'vehicle'
                        
                        # Verificar si está en polígono
                        in_polygon = is_point_in_polygon(centroid, self.detection_zone)
                        
                        # Actualizar o crear objeto en el tracker
                        internal_id = self.object_tracker.update_or_create_object(
                            yolo_track_id=yolo_track_id,
                            centroid=centroid,
                            bbox=(x1, y1, x2, y2),
                            object_type=object_type,
                            frame_count=frame_count,
                            in_polygon=in_polygon
                        )
                        
                        # Verificar si debe enviar alerta de movimiento
                        if self.object_tracker.check_movement_alert(internal_id):
                            movement_alerts_to_send.add(internal_id)
                        
                        # Solo dibujar si save_video está activo
                        #if self.save_video:
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
                            if calculate_intersection([x1, y1, x2, y2], ruma.initial_mask):
                                # Registrar que este objeto intersecta con esta ruma
                                if internal_id not in object_intersections:
                                    object_intersections[internal_id] = set()
                                object_intersections[internal_id].add(ruma_id)
                                
                                # Agregar a objects_per_ruma
                                if ruma_id not in objects_per_ruma:
                                    objects_per_ruma[ruma_id] = set()
                                objects_per_ruma[ruma_id].add(internal_id)

        return frame, movement_alerts_to_send, objects_per_ruma, object_intersections

    def process_segmentation(self, frame, frame_count, objects_per_ruma, object_intersections):
        """
        Procesa la segmentación de rumas.
        
        Args:
            objects_per_ruma: Dict[ruma_id -> Set[internal_ids]] de objetos en cada ruma
            object_intersections: Dict[internal_id -> Set[ruma_ids]] de rumas que toca cada objeto
            
        Returns:
            - frame: Frame con dibujos
            - interaction_alerts_to_send: Set[(internal_id, ruma_id)] de interacciones confirmadas
            - variation_alerts_to_send: Set[ruma_id] de rumas con variación >= 15%
        """
        result_seg = self.model_seg(frame, conf=0.5, verbose=False)
        
        interaction_alerts_to_send = set()
        variation_alerts_to_send = set()
        max_frames_without_interaction = 15

        if result_seg and len(result_seg) > 0:
            for r in result_seg:
                if r.masks is not None:
                    masks = r.masks.xy

                    for mask in masks:
                        centroid_x = int(np.mean([p[0] for p in mask]))
                        centroid_y = int(np.mean([p[1] for p in mask]))
                        centroid = (centroid_x, centroid_y)

                        # Solo procesar rumas dentro de la zona de detección
                        if not is_point_in_polygon(centroid, self.detection_zone):
                            continue

                        # Buscar ruma existente más cercana 
                        closest_ruma_id, distance = self.tracker.find_closest_ruma(centroid)

                        if closest_ruma_id is not None:
                            # Actualizar ruma existente
                            self.tracker.update_ruma(closest_ruma_id, mask, frame_count)
                            ruma = self.tracker.rumas[closest_ruma_id]

                            # Solo dibujar si save_video está activo
                            #if self.save_video:
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask.astype(np.int32)], self.RUMA_COLOR)
                            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                            # Verificar interacciones usando ObjectTracker
                            is_interacting = closest_ruma_id in objects_per_ruma and len(objects_per_ruma[closest_ruma_id]) > 0
                            
                            # Actualizar estado de interacción de la ruma
                            if ruma.was_interacting and not is_interacting:
                                # Terminó la interacción - verificar variación
                                if ruma.should_send_variation_alert():
                                    variation_alerts_to_send.add(closest_ruma_id)
                            
                            ruma.was_interacting = is_interacting

                            if is_interacting:
                                ruma.frames_without_interaction = 0
                                ruma.last_stable_percentage = ruma.percentage
                                
                                # Verificar cada objeto que intersecta con esta ruma
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

                            # Solo mostrar texto si save_video está activo
                            if self.save_video:
                                label_text = f"R{ruma.id} | {display_percentage:.1f}%"
                                frame = put_text_with_background(
                                    frame, label_text, ruma.label_position,
                                    font_scale=0.6, color=self.TEXT_COLOR_WHITE
                                )

                        else:
                            # Posible nueva ruma - agregar como candidata
                            self.tracker.add_candidate_ruma(mask, centroid, frame_count, frame.shape, self.transformer)

        # Limpiar candidatas antiguas (más de 100 frames sin confirmación)
        self.tracker.clean_old_candidates(frame_count)

        return frame, interaction_alerts_to_send, variation_alerts_to_send

    def process_frame(self, frame, frame_count, fps):
        """Procesa un frame completo"""
        # Verificar que el frame es válido
        if frame is None or frame.size == 0:
            print(f"[WARN] Frame {frame_count} inválido o vacío, saltando...")
            # Retornar frame negro del tamaño esperado si es posible
            if hasattr(self, '_last_valid_frame_shape'):
                return np.zeros(self._last_valid_frame_shape, dtype=np.uint8)
            else:
                return np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Guardar shape del último frame válido
        self._last_valid_frame_shape = frame.shape
        
        frame_with_drawings = frame.copy()

        # Procesar detecciones
        frame_with_drawings, movement_alerts, objects_per_ruma, object_intersections = self.process_detections(
            frame_with_drawings, frame_count
        )

        # Procesar segmentación
        frame_with_drawings, interaction_alerts, variation_alerts = self.process_segmentation(
            frame_with_drawings, frame_count, objects_per_ruma, object_intersections
        )

        # Solo dibujar zona y estado si save_video está activo
        if self.save_video:
            # Determinar estados para visualización
            has_movement = len(movement_alerts) > 0
            has_interaction = len(interaction_alerts) > 0
            has_variation = len(variation_alerts) > 0
            
            frame_with_drawings = draw_zone_and_status(
                frame_with_drawings,
                self.detection_zone,
                has_movement,
                has_interaction,
                has_variation,
                TEXT_COLOR_RED=self.TEXT_COLOR_RED,
                TEXT_COLOR_GREEN=self.TEXT_COLOR_GREEN
            )

        # === ENVIAR ALERTAS ===
        
        # 1. Alertas de movimiento en zona
        for internal_id in movement_alerts:
            ruma_data = RumaInfo(
                id=None,
                percent=None,
                centroid=None,
                radius=None,
                centroid_homographic=None,
                radius_homographic=None
            )
            
            save_alert(
                alert_type='movimiento_zona',
                ruma_data=ruma_data,
                frame=frame_with_drawings,
                frame_count=frame_count,
                fps=fps,
                camera_sn=self.camera_sn,
                enterprise=self.enterprise,
                api_url=self.api_url,
                send=self.send,
                save=self.save,
                ruma_summary=self.tracker.ruma_summary,
                frame_shape=frame.shape,
                detection_zone=self.detection_zone
            )
        
        # 2. Alertas de interacción con rumas
        for (internal_id, ruma_id) in interaction_alerts:
            if ruma_id in self.tracker.rumas:
                ruma = self.tracker.rumas[ruma_id]
                ruma_data = RumaInfo(
                    id=ruma.id,
                    percent=ruma.last_stable_percentage,
                    centroid=ruma.centroid,
                    radius=ruma.radius,
                    centroid_homographic=ruma.centroid_homographic,
                    radius_homographic=ruma.radius_homographic
                )
                
                save_alert(
                    alert_type='interaccion_rumas',
                    ruma_data=ruma_data,
                    frame=frame_with_drawings,
                    frame_count=frame_count,
                    fps=fps,
                    camera_sn=self.camera_sn,
                    enterprise=self.enterprise,
                    api_url=self.api_url,
                    send=self.send,
                    save=self.save,
                    ruma_summary=self.tracker.ruma_summary,
                    frame_shape=frame.shape,
                    detection_zone=self.detection_zone
                )
        
        # 3. Alertas de variación de rumas
        for ruma_id in variation_alerts:
            if ruma_id in self.tracker.rumas:
                ruma = self.tracker.rumas[ruma_id]
                ruma_data = RumaInfo(
                    id=ruma.id,
                    percent=ruma.percentage,
                    centroid=ruma.centroid,
                    radius=ruma.radius,
                    centroid_homographic=ruma.centroid_homographic,
                    radius_homographic=ruma.radius_homographic
                )
                
                save_alert(
                    alert_type='variacion_rumas',
                    ruma_data=ruma_data,
                    frame=frame_with_drawings,
                    frame_count=frame_count,
                    fps=fps,
                    camera_sn=self.camera_sn,
                    enterprise=self.enterprise,
                    api_url=self.api_url,
                    send=self.send,
                    save=self.save,
                    ruma_summary=self.tracker.ruma_summary,
                    frame_shape=frame.shape,
                    detection_zone=self.detection_zone
                )

        # 4. Detectar nuevas rumas
        if self.tracker.new_ruma_created:
            ruma_id, frame_shape = self.tracker.new_ruma_created
            ruma = self.tracker.rumas[ruma_id]


            if not hasattr(self, '_new_ruma_queue'):
                self._new_ruma_queue = []
            
            self._new_ruma_queue.append((ruma_id, frame_shape))
            
            # Procesar SOLO una ruma nueva por frame
            if len(self._new_ruma_queue) > 0:
                ruma_id, frame_shape = self._new_ruma_queue.pop(0)
                ruma = self.tracker.rumas[ruma_id]
            ruma_data = RumaInfo(
                id=ruma.id,
                percent=100.0,
                centroid=ruma.centroid,
                radius=ruma.radius,
                centroid_homographic=ruma.centroid_homographic,
                radius_homographic=ruma.radius_homographic
            )

            save_alert(
                alert_type='nueva_ruma',
                ruma_data=ruma_data,
                frame=frame_with_drawings,
                frame_count=frame_count,
                fps=fps,
                camera_sn=self.camera_sn,
                enterprise=self.enterprise,
                api_url=self.api_url,
                send=self.send,
                save=self.save,
                ruma_summary=self.tracker.ruma_summary,
                frame_shape=frame.shape,
                detection_zone=self.detection_zone
            )

            self.tracker.new_ruma_created = None
        
        # Limpiar objetos antiguos del tracker
        self.object_tracker.cleanup_old_objects(frame_count)

        return frame_with_drawings