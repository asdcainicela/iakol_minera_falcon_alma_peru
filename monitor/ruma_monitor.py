import os
import cv2
import time
import json
import base64
import torch
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, deque

#--------- import de la clase ruma data -------#
from monitor.ruma_data import RumaData
from utils.geometry import is_point_in_polygon, calculate_intersection
from alerts.alert_manager import save_alert
from utils.paths import setup_alerts_folder
#-----------#

class RumaMonitor:
    def __init__(self, model_det_path, model_seg_path, detection_zone, camera_sn):
        """
        Inicializa el monitor de rumas

        Args:
            model_det_path: Ruta al modelo de detección
            model_seg_path: Ruta al modelo de segmentación
            detection_zone: Polígono que define la zona de detección
            camera_sn: Número de serie de la cámara
        """
        self.api_url = "https://fn-alma-mina.azurewebsites.net/api/alert"
        self.model_det = YOLO(model_det_path)
        self.model_seg = YOLO(model_seg_path)
        self.detection_zone = detection_zone
        self.camera_sn = camera_sn
        self.enterprise = None # Added enterprise attribute

        # Tracking de rumas
        self.rumas = {}  # ruma_id: RumaData
        self.next_ruma_id = 1
        self.candidate_rumas = {}  # Para validar nuevas rumas por 10 frames

        # Estados de alertas
        self.object_in_zone = False
        self.object_interacting = False
        self.ruma_variation = False
        self.last_interaction_frame = {}  # Para controlar cuando mostrar porcentajes

        # Configuración de colores
        self.RUMA_COLOR = (0, 255, 0)
        self.PERSON_COLOR = (255, 255, 0)
        self.VEHICLE_COLOR = (0, 0, 255)
        self.TEXT_COLOR_WHITE = (255, 255, 255)
        self.TEXT_COLOR_GREEN = (0, 255, 0)
        self.TEXT_COLOR_RED = (0, 0, 255)

        # Crear carpeta de alertas
        self.setup_alerts_folder =setup_alerts_folder()

        #
        self.ruma_summary = {}
        self.new_ruma_created = None  #  marca si hay nueva ruma
    
    def put_text_with_background(self, img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale=0.4, color=(255,255,255), thickness=1,
                                bg_color=(0,0,0), bg_alpha=0.6):
        """Coloca texto con fondo semitransparente para mejor legibilidad"""
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = position
        bg_img = img.copy()
        padding = 5

        cv2.rectangle(bg_img, (x-padding, y-text_height-padding),
                     (x+text_width+padding, y+padding), bg_color, -1)

        overlay = cv2.addWeighted(bg_img, bg_alpha, img, 1-bg_alpha, 0)
        cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)

        return overlay

    def find_closest_ruma(self, centroid, max_distance=50): # max_distance=100
        """Encuentra la ruma más cercana a un centroide dado"""
        min_distance = float('inf')
        closest_ruma = None

        for ruma_id, ruma in self.rumas.items():
            if not ruma.is_active:
                continue

            dist = ((centroid[0] - ruma.centroid[0])**2 +
                   (centroid[1] - ruma.centroid[1])**2)**0.5

            if dist < min_distance and dist < max_distance:
                min_distance = dist
                closest_ruma = ruma_id

        return closest_ruma, min_distance

    def add_candidate_ruma(self, mask, centroid, frame_count, frame_shape):
      candidate_key = f"candidate_{centroid[0]}_{centroid[1]}"

      if candidate_key not in self.candidate_rumas:
          self.candidate_rumas[candidate_key] = {
              'mask': mask,
              'centroid': centroid,
              'area': cv2.contourArea(mask.astype(np.int32)),
              'first_frame': frame_count,
              'confirmations': 1
          }
      else:
          self.candidate_rumas[candidate_key]['confirmations'] += 1

          if self.candidate_rumas[candidate_key]['confirmations'] >= 6:
              ruma_id = self.next_ruma_id
              area = cv2.contourArea(mask.astype(np.int32))
              # usamos RumaData del import ruma_data.py
              # new_ruma = self.RumaData(ruma_id, mask, area, centroid)
              new_ruma = RumaData(ruma_id, mask, area, centroid)
              self.rumas[ruma_id] = new_ruma
              print(f"Nueva ruma creada: ID {ruma_id}")

              self.store_ruma_summary(ruma_id, mask, centroid, frame_shape)

              self.next_ruma_id += 1
              del self.candidate_rumas[candidate_key]

    def store_ruma_summary(self, ruma_id, mask, centroid, frame_shape):
        """Guarda resumen de la ruma (solo metadatos, no imagen aún)"""
        distances = [np.linalg.norm(np.array(centroid) - np.array(p)) for p in mask]
        radius = float(np.mean(distances))

        self.ruma_summary[ruma_id] = {
            'centroid': tuple(map(int, centroid)),
            'radius': round(radius, 2)
        }

        print(f" Ruma {ruma_id} almacenada en resumen:")
        print(self.ruma_summary[ruma_id])

        #  Guardar frame_shape y ruma_id para generar imagen luego (en save_alert2)
        self.new_ruma_created = (ruma_id, frame_shape)

    def update_ruma(self, ruma_id, mask, frame_count):
        """Actualiza los datos de una ruma existente"""
        ruma = self.rumas[ruma_id]
        ruma.current_area = cv2.contourArea(mask.astype(np.int32))
        ruma.percentage = (ruma.current_area / ruma.initial_area) * 100
        ruma.last_seen_frame = frame_count

        # Actualizar centroide
        ruma.centroid = (int(np.mean([p[0] for p in mask])),
                        int(np.mean([p[1] for p in mask])))

        # Si la ruma llega a 10% o menos, marcarla como inactiva
        if ruma.percentage <= 10:
            ruma.is_active = False
            print(f"Ruma {ruma_id} eliminada (porcentaje: {ruma.percentage:.1f}%)")

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
                        frame = self.put_text_with_background(
                            frame, label, (x1, y1 - 5),
                            color=self.TEXT_COLOR_WHITE, font_scale=0.6
                        )

                        # Verificar si está en la zona
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if is_point_in_polygon((center_x, center_y), self.detection_zone):
                            object_in_zone = True

                        # Verificar interacción con rumas
                        for ruma_id, ruma in self.rumas.items():
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
                        closest_ruma_id, distance = self.find_closest_ruma(centroid)

                        if closest_ruma_id is not None:
                            # Actualizar ruma existente
                            self.update_ruma(closest_ruma_id, mask, frame_count)
                            ruma = self.rumas[closest_ruma_id]

                            # Dibujar la ruma
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [mask.astype(np.int32)], self.RUMA_COLOR)
                            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                            # --------  Lógica mejorada para mostrar porcentaje --------- #
                            # 1. Detectar interacción actual
                            is_interacting = closest_ruma_id in rumas_interacting
                            #print("is_interacting =", is_interacting, "type:", type(is_interacting))

                            # Detectar fin de interacción (flanco de bajada)
                            draw_object_interacting = is_interacting

                            if ruma.was_interacting and not is_interacting:
                              object_interaction_ended = True
                              draw_object_interacting = True

                            if is_interacting and not ruma.was_interacting:
                              object_interacting = True
                              draw_object_interacting = True


                            # 3. Guardar estado actual para el siguiente frame
                            ruma.was_interacting = is_interacting

                            if is_interacting:
                                ruma.frames_without_interaction = 0
                                ruma.last_stable_percentage = ruma.percentage  # Reiniciar al interactuar
                                #draw_object_interacting = True
                            else:
                                ruma.frames_without_interaction += 1
                                # Mientras no se llega a 30 frames, guarda el mayor porcentaje
                                if ruma.frames_without_interaction < max_frames_without_interaction: #30
                                    ruma.last_stable_percentage = max(ruma.last_stable_percentage, ruma.percentage)
                                #draw_object_interacting = False

                            # Limita todo a máximo 100%
                            ruma.percentage = min(ruma.percentage, 100)
                            ruma.last_stable_percentage = min(ruma.last_stable_percentage, 100)

                            #if ruma.frames_without_interaction == max_frames_without_interaction: #corregir
                            # object_interacting = True # corregir


                            # Decide qué mostrar
                            if ruma.frames_without_interaction >= max_frames_without_interaction:
                                display_percentage = ruma.last_stable_percentage
                                draw_ruma_variation = False
                            else:
                                display_percentage = ruma.percentage
                                draw_ruma_variation = True

                            #--------                                          -------#

                            # Detectar variación significativa
                            #if display_percentage < 95:
                            #    ruma_variation = True

                            # Mostrar ID y porcentaje usando posición fija del label
                            label_text = f"R{ruma.id} | {display_percentage:.1f}%"
                            frame = self.put_text_with_background(
                                frame, label_text, ruma.label_position,  # ← Usar posición fija
                                font_scale=0.6, color=self.TEXT_COLOR_WHITE
                            )

                        else:
                            # Posible nueva ruma - agregar como candidata
                            #self.add_candidate_ruma(mask, centroid, frame_count)
                            self.add_candidate_ruma(mask, centroid, frame_count, frame.shape)

                            #centroide_creado = self.add_candidate_ruma(mask, centroid, frame_count)

                            #if centroide_creado is not None:
                            #    print("Ruma confirmada con centroide:", centroide_creado)
                            #print(f"el nuevo centroide es {centroid}")
                            #print(f" el centroide creado es_{centroid[0]}_{centroid[1]}")

        # Limpiar candidatas antiguas (más de 100 frames sin confirmación)
        to_remove = []
        for key, candidate in self.candidate_rumas.items():
            if frame_count - candidate['first_frame'] > 100:
                to_remove.append(key)

        for key in to_remove:
            del self.candidate_rumas[key]

        ruma_variation = object_interaction_ended ## tomar screen ultima parte

        return frame, ruma_variation, object_interacting, draw_ruma_variation, draw_object_interacting

    def draw_zone_and_status(self, frame, draw_object_in_zone, object_interacting, draw_ruma_variation):
        """Dibuja la zona de detección y el estado de las alertas"""
        width = frame.shape[1]

        # Dibujar zona de detección
        pts = self.detection_zone.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # Textos de estado
        text_y_start = 50

        zone_text = "Movimiento en la zona" if draw_object_in_zone else "Zona despejada"
        zone_color = self.TEXT_COLOR_RED if draw_object_in_zone else self.TEXT_COLOR_GREEN
        frame = self.put_text_with_background(
            frame, zone_text, (width - 650, text_y_start),
            color=zone_color, font_scale=1.5
        )

        interact_text = "Interaccion con las rumas" if object_interacting else "Sin interacciones"
        interact_color = self.TEXT_COLOR_RED if object_interacting else self.TEXT_COLOR_GREEN
        frame = self.put_text_with_background(
            frame, interact_text, (width - 650, text_y_start + 60),
            color=interact_color, font_scale=1.5
        )

        variation_text = "Variacion en las rumas" if draw_ruma_variation else "Rumas en reposo"
        variation_color = self.TEXT_COLOR_RED if draw_ruma_variation else self.TEXT_COLOR_GREEN
        frame = self.put_text_with_background(
            frame, variation_text, (width - 650, text_y_start + 120),
            color=variation_color, font_scale=1.5
        )

        return frame

    def process_frame(self, frame, frame_count, fps):
        """Procesa un frame completo"""
        frame_with_drawings = frame.copy()

        # Procesar detecciones
        frame_with_drawings, object_in_zone, rumas_interacting = self.process_detections(
            frame_with_drawings, frame_count
        )

        # Procesar segmentación
        frame_with_drawings, ruma_variation, object_interacting, draw_ruma_variation, draw_object_interacting = self.process_segmentation(
            frame_with_drawings, frame_count, rumas_interacting
        )

        # Determinar estados de alerta
        #object_interacting = len(rumas_interacting) > 0

        # Dibujar zona y estado
        frame_with_drawings = self.draw_zone_and_status(
            frame_with_drawings, object_in_zone, draw_ruma_variation, draw_object_interacting
        )

        # Guardar alertas si hay cambios de estado
        current_alerts = {
            'movement': object_in_zone,
            'interaction': object_interacting,
            'variation': ruma_variation
        }

        previous_alerts = {
            'movement': self.object_in_zone,
            'interaction': self.object_interacting,
            'variation': self.ruma_variation
        }

        # Guardar alertas solo cuando se activan (cambio de False a True)
        for alert_type, current_state in current_alerts.items():
            if current_state and not previous_alerts[alert_type]:
                alert_names = {
                    'movement': 'movimiento_zona',
                    'interaction': 'interaccion_rumas',
                    'variation': 'variacion_rumas'#,
                    #'new': 'nueva_ruma'
                }
                #self.save_alert(alert_names[alert_type], frame_with_drawings, frame_count, fps)
                #self.save_alert2(alert_names[alert_type], frame_with_drawings, frame_count, fps)
                save_alert(
                    alert_type=alert_names[alert_type],
                    frame=frame_with_drawings,
                    frame_count=frame_count,
                    fps=fps,
                    camera_sn=self.camera_sn,
                    enterprise='alma',
                    api_url=self.api_url,
                    send=False,
                    save=True,
                    ruma_summary=self.ruma_summary ,
                    frame_shape=frame.shape,
                    detection_zone = self.detection_zone
                )



        # Actualizar estados
        self.object_in_zone = object_in_zone
        self.object_interacting = object_interacting
        self.ruma_variation = ruma_variation

        return frame_with_drawings

