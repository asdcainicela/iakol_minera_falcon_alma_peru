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
from alerts.alert_info import RumaInfo # Dataclass para almacenar datos de rumas

#-----------#

class RumaMonitor:
    def __init__(self, model_det_path, model_seg_path, detection_zone, camera_sn, api_url, transformer):
        """
        Inicializa el monitor de rumas

        Args:
            model_det_path: Ruta al modelo de detección
            model_seg_path: Ruta al modelo de segmentación
            detection_zone: Polígono que define la zona de detección
            camera_sn: Número de serie de la cámara
        """
        self.api_url = api_url 
        self.model_det = YOLO(model_det_path)
        self.model_seg = YOLO(model_seg_path)
        self.detection_zone = detection_zone
        self.camera_sn = camera_sn
        self.enterprise = 'alma' # Added enterprise attribute

        # Tracking de rumas
        self.tracker = RumaTracker()

        # Estados de alertas
        self.object_in_zone = False
        self.object_interacting = False
        self.ruma_variation = False
        self.new_variation = False
        self.last_interaction_frame = {}  # Para controlar cuando mostrar porcentajes

        # Configuración de colores
        self.RUMA_COLOR = (0, 255, 0)
        self.PERSON_COLOR = (255, 255, 0)
        self.VEHICLE_COLOR = (0, 0, 255)
        self.TEXT_COLOR_WHITE = (255, 255, 255)
        self.TEXT_COLOR_GREEN = (0, 255, 0)
        self.TEXT_COLOR_RED = (0, 0, 255)

        self.transformer = transformer

        # Envio de datos
        self.send = False # Envio de datos a la nube
        self.save = True # Guardado de datos local

    def process_detections(self, frame, frame_count):
        """Procesa las detecciones de personas y vehículos"""
        rumas_interacting = set()
        object_in_zone = False

        result_det = self.model_det.track(frame, conf=0.5, persist=True, verbose=False)

        if (result_det is not None) and len(result_det) > 0:
            boxes = result_det[0].boxes

            if (boxes is not None) and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > 0.5:
                        color = self.PERSON_COLOR if cls == 0 else self.VEHICLE_COLOR
                        label = 'persona' if cls == 0 else 'maquinaria'

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        frame = put_text_with_background(
                            frame, label, (x1, y1 - 5),
                            color=self.TEXT_COLOR_WHITE, font_scale=0.6
                        )

                        # Verificar si está en la zona
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if is_point_in_polygon((center_x, center_y), self.detection_zone):
                            object_in_zone = True

                        # Verificar interacción con rumas
                        for ruma_id, ruma in self.tracker.rumas.items():
                            if not ruma.is_active:
                                continue
                            if calculate_intersection([x1, y1, x2, y2], ruma.initial_mask):
                                rumas_interacting.add(ruma_id)
                                self.last_interaction_frame[ruma_id] = frame_count

        return frame, object_in_zone, rumas_interacting

    def process_segmentation(self, frame, frame_count, rumas_interacting):
        """Procesa la segmentación de rumas"""
        result_seg = self.model_seg(frame, conf=0.5, verbose=False)
        ruma_variation = False
        object_interacting = False  # Initialize object_interacting
        object_interaction_ended = False # ultimo frame
        max_frames_without_interaction = 15
        # Draw zone
        draw_object_interacting = False
        draw_ruma_variation = False
        ruma_with_variation = None  # Para guardar qué ruma tuvo variación
        #---

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

                            # Dibujar la ruma
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask.astype(np.int32)], self.RUMA_COLOR)
                            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                            # --------  Lógica mejorada para mostrar porcentaje --------- #
                            # 1. Detectar interacción actual
                            is_interacting = closest_ruma_id in rumas_interacting

                            # Detectar fin de interacción (flanco de bajada)
                            draw_object_interacting = is_interacting

                            if ruma.was_interacting and not is_interacting:
                              object_interaction_ended = True
                              draw_object_interacting = True
                              ruma_with_variation = ruma  # Guardar la ruma que tuvo variación

                            if is_interacting and not ruma.was_interacting:
                              object_interacting = True
                              draw_object_interacting = True

                            # 3. Guardar estado actual para el siguiente frame
                            ruma.was_interacting = is_interacting

                            if is_interacting:
                                ruma.frames_without_interaction = 0
                                ruma.last_stable_percentage = ruma.percentage  # Reiniciar al interactuar
                            else:
                                ruma.frames_without_interaction += 1
                                # Mientras no se llega a 30 frames, guarda el mayor porcentaje
                                if ruma.frames_without_interaction < max_frames_without_interaction: #30
                                    ruma.last_stable_percentage = max(ruma.last_stable_percentage, ruma.percentage)

                            # Limita todo a máximo 100%
                            ruma.percentage = min(ruma.percentage, 100)
                            ruma.last_stable_percentage = min(ruma.last_stable_percentage, 100)

                            # Decide qué mostrar
                            if ruma.frames_without_interaction >= max_frames_without_interaction:
                                display_percentage = ruma.last_stable_percentage
                                draw_ruma_variation = False
                            else:
                                display_percentage = ruma.percentage
                                draw_ruma_variation = True

                            # Mostrar ID y porcentaje usando posición fija del label
                            label_text = f"R{ruma.id} | {display_percentage:.1f}%"
                            frame = put_text_with_background(
                                frame, label_text, ruma.label_position,  # ← Usar posición fija
                                font_scale=0.6, color=self.TEXT_COLOR_WHITE
                            )

                        else:
                            # Posible nueva ruma - agregar como candidata
                            self.tracker.add_candidate_ruma(mask, centroid, frame_count, frame.shape, self.transformer)

        # Limpiar candidatas antiguas (más de 100 frames sin confirmación)
        self.tracker.clean_old_candidates(frame_count)

        ruma_variation = object_interaction_ended ## tomar screen ultima parte

        return frame, ruma_variation, object_interacting, draw_ruma_variation, draw_object_interacting, ruma_with_variation

    def process_frame(self, frame, frame_count, fps):
        """Procesa un frame completo"""
        frame_with_drawings = frame.copy()

        # Procesar detecciones
        frame_with_drawings, object_in_zone, rumas_interacting = self.process_detections(
            frame_with_drawings, frame_count
        )

        # Procesar segmentación
        frame_with_drawings, ruma_variation, object_interacting, draw_ruma_variation, draw_object_interacting, ruma_with_variation = self.process_segmentation(
            frame_with_drawings, frame_count, rumas_interacting
        )

        # Dibujar zona y estado
        frame_with_drawings = draw_zone_and_status(
            frame_with_drawings,
            self.detection_zone,
            object_in_zone,
            draw_object_interacting,
            draw_ruma_variation,
            TEXT_COLOR_RED=self.TEXT_COLOR_RED,
            TEXT_COLOR_GREEN=self.TEXT_COLOR_GREEN
        )

        # Guardar alertas si hay cambios de estado
        current_alerts = {
            'movement': object_in_zone,
            'interaction': object_interacting,
            'variation': ruma_variation,
            'new': self.new_variation
        }

        previous_alerts = {
            'movement': self.object_in_zone,
            'interaction': self.object_interacting,
            'variation': self.ruma_variation,
            'new': self.new_variation
        }

        # Detectar nuevas rumas
        if self.tracker.new_ruma_created:
            ruma_id, frame_shape = self.tracker.new_ruma_created
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
                frame=None,
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

        # Guardar alertas cuando se activan (cambio de False a True)
        for alert_type, current_state in current_alerts.items():
            if current_state and not previous_alerts[alert_type]:
                alert_names = {
                    'movement': 'movimiento_zona',
                    'interaction': 'interaccion_rumas',
                    'variation': 'variacion_rumas',
                    'new': 'nueva_ruma'
                }

                ruma_data = None

                if alert_type in ['movement', 'interaction']:
                    ruma_data = RumaInfo(
                        id=None,
                        percent=None,
                        centroid=None,
                        radius=None,
                        centroid_homographic=None,
                        radius_homographic=None
                    )

                elif alert_type == 'variation':
                    if ruma_with_variation is not None:
                        ruma_data = RumaInfo(
                            id=ruma_with_variation.id,
                            percent=ruma_with_variation.last_stable_percentage,
                            centroid=ruma_with_variation.centroid,
                            radius=ruma_with_variation.radius,
                            centroid_homographic=ruma_with_variation.centroid_homographic,
                            radius_homographic=ruma_with_variation.radius_homographic
                        )
                    else:
                        ruma_data = RumaInfo(
                            id=None,
                            percent=None,
                            centroid=None,
                            radius=None,
                            centroid_homographic=None,
                            radius_homographic=None
                        )

                if ruma_data is not None:
                    save_alert(
                        alert_type=alert_names[alert_type],
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

        # Actualizar estados
        self.object_in_zone = object_in_zone
        self.object_interacting = object_interacting
        self.ruma_variation = ruma_variation
        self.new_variation = False  # Reset después de procesar

        return frame_with_drawings
