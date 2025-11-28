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
                 api_url, transformer, save_video=False,
                 segmentation_interval_idle=100,
                 segmentation_interval_active=10,
                 activity_cooldown_frames=200,
                 detection_skip_idle=3):
        """
        Inicializa el monitor de rumas con optimizaciones de rendimiento

        Args:
            detection_skip_idle: Cada cuántos frames detectar en modo SLEEP (default: 3)
        """
        self.api_url = api_url 
        
        # NUEVO: Detectar si son modelos TensorRT y ajustar parámetros
        self.is_tensorrt_det = model_det_path.endswith('.engine')
        self.is_tensorrt_seg = model_seg_path.endswith('.engine')
        
        # Cargar modelos
        self.model_det = YOLO(model_det_path)
        self.model_seg = YOLO(model_seg_path)
        
        # NUEVO: Tamaño de imagen según tipo de modelo
        self.det_imgsz = 1024 if self.is_tensorrt_det else 640
        self.seg_imgsz = 1024 if self.is_tensorrt_seg else 640
        
        print(f"[RumaMonitor] Modelos cargados:")
        print(f"  - Detección: {'TensorRT' if self.is_tensorrt_det else 'PyTorch'} @ {self.det_imgsz}px")
        print(f"  - Segmentación: {'TensorRT' if self.is_tensorrt_seg else 'PyTorch'} @ {self.seg_imgsz}px")

        self.detection_zone = detection_zone
        self.camera_sn = camera_sn
        self.enterprise = 'alma'
        self.save_video = save_video

        # Tracking de rumas
        self.tracker = RumaTracker()
        
        # Tracking de objetos (personas/vehículos)
        self.object_tracker = ObjectTracker(
            interaction_threshold=40,
            max_distance_match=150  # Aumentado para cámara PTZ con viento
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
        self.send = True
        self.save = False
        
        # === SEGMENTACIÓN CONDICIONAL ===
        self.segmentation_interval_idle = segmentation_interval_idle
        self.segmentation_interval_active = segmentation_interval_active
        self.activity_cooldown_frames = activity_cooldown_frames
        
        # === DETECCIÓN CONDICIONAL (NUEVO) ===
        self.detection_skip_idle = detection_skip_idle
        self.sleep_threshold_frames = 600  # 600 frames sin actividad = SLEEP (~20s @ 30fps)
        
        # Estado del sistema (3 niveles ahora)
        self.system_state = "IDLE"  # Iniciar en IDLE para detectar rumas iniciales
        self.frames_without_activity = 0
        self.last_segmentation_frame = -999  # Forzar segmentación inmediata
        self.cached_segmentation_data = None
        self.initial_scan_complete = False  # Flag para scan inicial
        
        # Cache de detección (para skip frames)
        self.last_detection_result = {
            'objects': [],
            'frame_number': -1
        }
        
        # === CROP DE POLÍGONO (NUEVO) ===
        self._compute_polygon_bbox()
        
        print(f"[RumaMonitor] Sistema optimizado iniciado:")
        print(f"  - Segmentación IDLE: cada {segmentation_interval_idle} frames")
        print(f"  - Segmentación ACTIVE: cada {segmentation_interval_active} frames")
        print(f"  - Detección SLEEP: cada {detection_skip_idle} frames")
        print(f"  - Cooldown ACTIVE→IDLE: {activity_cooldown_frames} frames")
        print(f"  - Cooldown IDLE→SLEEP: {self.sleep_threshold_frames} frames")
        print(f"  - Crop polígono: {self.crop_bbox}")

    def _compute_polygon_bbox(self):
        """Calcula bounding box del polígono para crop optimizado"""
        points = np.array(self.detection_zone)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Agregar padding del 10%
        padding_x = int((x_max - x_min) * 0.1)
        padding_y = int((y_max - y_min) * 0.1)
        
        # CRÍTICO: Limitar a las dimensiones del frame
        self.crop_bbox = (
            max(0, x_min - padding_x),
            max(0, y_min - padding_y),
            min(1920, x_max + padding_x),  # NUEVO: Límite
            min(1080, y_max + padding_y)   # NUEVO: Límite
        )
        
        # Offset para ajustar coordenadas
        self.crop_offset = (self.crop_bbox[0], self.crop_bbox[1])
        
        print(f"[RumaMonitor] Crop polígono: {self.crop_bbox}")
    
    def _crop_to_polygon(self, frame):
        """Cropea frame al bounding box del polígono"""
        x1, y1, x2, y2 = self.crop_bbox
        h, w = frame.shape[:2]
        
        # Asegurar que no se salga de bounds
        x2 = min(x2, w)
        y2 = min(y2, h)
        
        return frame[y1:y2, x1:x2]

    def _adjust_bbox_to_full_frame(self, bbox):
        """Ajusta coordenadas del crop al frame completo"""
        x1, y1, x2, y2 = bbox
        offset_x, offset_y = self.crop_offset
        return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)

    def _should_skip_detection(self, frame_count):
        """Determina si debe skippear detección según estado"""
        # DURANTE SCAN INICIAL, NUNCA SKIP (queremos detectar todo)
        if not self.initial_scan_complete:
            return False
        
        if self.system_state == "SLEEP":
            # En SLEEP, detectar cada N frames
            return frame_count % self.detection_skip_idle != 0
        
        # En IDLE y ACTIVE, siempre detectar
        return False

    def process_detections(self, frame, frame_count):
        """
        Procesa las detecciones de personas y vehículos CON OPTIMIZACIONES.
        
        CAMBIOS:
        - Usa .predict() en lugar de .track() (más rápido)
        - Cropea frame a polígono antes de YOLO
        - Skip inteligente en modo SLEEP
        """
        movement_alerts_to_send = set()
        objects_per_ruma = {}
        object_intersections = {}
        has_activity_in_zone = False

        # === SKIP DETECCIÓN SI ESTAMOS EN SLEEP ===
        if self._should_skip_detection(frame_count):
            # Reutilizar última detección conocida
            return (frame, movement_alerts_to_send, objects_per_ruma, 
                    object_intersections, has_activity_in_zone)

        # === CROP FRAME A POLÍGONO ===
        frame_crop = self._crop_to_polygon(frame)

        # === USAR .predict() EN LUGAR DE .track() ===
        #result_det = self.model_det.predict(frame_crop, conf=0.5, verbose=False)
        result_det = self.model_det.predict(
            frame_crop, 
            conf=0.7, 
            verbose=False,
            imgsz=self.det_imgsz  # NUEVO: usar tamaño correcto
        )   

        if (result_det is not None) and len(result_det) > 0:
            boxes = result_det[0].boxes

            if (boxes is not None) and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > 0.5:
                        # Ajustar coordenadas al frame completo
                        x1, y1, x2, y2 = self._adjust_bbox_to_full_frame((x1, y1, x2, y2))

                        # Calcular centroide
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        centroid = (center_x, center_y)
                        
                        # Tipo de objeto
                        object_type = 'person' if cls == 0 else 'vehicle'
                        
                        # Verificar si está en polígono
                        in_polygon = is_point_in_polygon(centroid, self.detection_zone)
                        
                        # Actualizar o crear objeto (SIN yolo_track_id, usamos None)
                        internal_id = self.object_tracker.update_or_create_object(
                            yolo_track_id=None,  # Ya no usamos YOLO tracking
                            centroid=centroid,
                            bbox=(x1, y1, x2, y2),
                            object_type=object_type,
                            frame_count=frame_count,
                            in_polygon=in_polygon
                        )
                        
                        # Detectar actividad en zona
                        if in_polygon:
                            has_activity_in_zone = True
                        
                        # Verificar si debe enviar alerta de movimiento
                        if self.object_tracker.check_movement_alert(internal_id):
                            movement_alerts_to_send.add(internal_id)
                        
                        # Solo dibujar si save_video está activo
                        if self.save_video:
                            color = self.PERSON_COLOR if cls == 0 else self.VEHICLE_COLOR
                            label = f'{object_type} ID:{internal_id}'
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            frame = put_text_with_background(
                                frame, label, (x1, y1 - 5),
                                color=self.TEXT_COLOR_WHITE, font_scale=0.6
                            )
                        
                        # Verificar interacción con rumas (SOLO si hay rumas activas)
                        if len(self.tracker.rumas) > 0:
                            for ruma_id, ruma in self.tracker.rumas.items():
                                if not ruma.is_active:
                                    continue
                                if calculate_intersection([x1, y1, x2, y2], ruma.initial_mask):
                                    if internal_id not in object_intersections:
                                        object_intersections[internal_id] = set()
                                    object_intersections[internal_id].add(ruma_id)
                                    
                                    if ruma_id not in objects_per_ruma:
                                        objects_per_ruma[ruma_id] = set()
                                    objects_per_ruma[ruma_id].add(internal_id)

        # === ACTUALIZAR ESTADO DEL SISTEMA (3 NIVELES) ===
        if has_activity_in_zone:
            self.frames_without_activity = 0
            
            if self.system_state in ["SLEEP", "IDLE"]:
                print(f"[RumaMonitor] Frame {frame_count}: TRANSICIÓN {self.system_state} → ACTIVE")
                self.system_state = "ACTIVE"
        else:
            self.frames_without_activity += 1
            
            # ACTIVE → IDLE (rápido, 200 frames)
            if self.system_state == "ACTIVE" and self.frames_without_activity >= self.activity_cooldown_frames:
                print(f"[RumaMonitor] Frame {frame_count}: TRANSICIÓN ACTIVE → IDLE")
                self.system_state = "IDLE"
            
            # IDLE → SLEEP (lento, 600 frames) - SOLO SI YA COMPLETÓ SCAN INICIAL
            elif (self.system_state == "IDLE" and 
                  self.initial_scan_complete and 
                  self.frames_without_activity >= self.sleep_threshold_frames):
                print(f"[RumaMonitor] Frame {frame_count}: TRANSICIÓN IDLE → SLEEP")
                self.system_state = "SLEEP"

        return frame, movement_alerts_to_send, objects_per_ruma, object_intersections, has_activity_in_zone

    def _use_cached_segmentation(self, frame, frame_count, objects_per_ruma, object_intersections):
        """Reutiliza la última segmentación conocida cuando se hace skip"""
        
        if self.cached_segmentation_data is None:
            print(f"[RumaMonitor] Frame {frame_count}: Cache vacío, forzando segmentación inicial")
            self.last_segmentation_frame = -999
            return frame, set(), set()
        
        # Dibujar rumas desde cache si save_video
        if self.save_video:
            for ruma_id, ruma in self.tracker.rumas.items():
                if not ruma.is_active:
                    continue
                    
                overlay = frame.copy()
                mask = ruma.initial_mask.astype(np.int32)
                cv2.fillPoly(overlay, [mask], self.RUMA_COLOR)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                
                label_text = f"R{ruma.id} | {ruma.percentage:.1f}%"
                frame = put_text_with_background(
                    frame, label_text, ruma.label_position,
                    font_scale=0.6, color=self.TEXT_COLOR_WHITE
                )
        
        # Verificar interacciones CON datos viejos de rumas
        interaction_alerts = set()
        for ruma_id in objects_per_ruma:
            for internal_id in objects_per_ruma[ruma_id]:
                intersecting = object_intersections.get(internal_id, set())
                should_alert, confirmed_ruma = self.object_tracker.update_interaction(
                    internal_id, intersecting
                )
                if should_alert:
                    interaction_alerts.add((internal_id, confirmed_ruma))
        
        return frame, interaction_alerts, set()

    def process_segmentation(self, frame, frame_count, objects_per_ruma, object_intersections):
        """Procesa la segmentación de rumas DE FORMA CONDICIONAL"""
        
        # === SCAN INICIAL AGRESIVO (primeros 30 frames) ===
        if not self.initial_scan_complete and frame_count <= 30:
            # Forzar segmentación cada 5 frames al inicio
            frames_since_last = frame_count - self.last_segmentation_frame
            if frames_since_last >= 3:
                print(f"[RumaMonitor] Frame {frame_count}: SCAN INICIAL (cada 5 frames)")
                self.last_segmentation_frame = frame_count
                # Continuar con segmentación normal abajo
            else:
                # Skip pero no usar cache (aún no hay rumas)
                return frame, set(), set()
            
            # Marcar scan completo al frame 30
            if frame_count >= 30:
                self.initial_scan_complete = True
                print(f"[RumaMonitor] SCAN INICIAL COMPLETADO - Rumas detectadas: {len(self.tracker.rumas)}")
                # Transición a SLEEP si no hay actividad
                if self.frames_without_activity > 20:
                    self.system_state = "SLEEP"
                    print(f"[RumaMonitor] Transición automática a SLEEP")
        else:
            # === LÓGICA NORMAL DESPUÉS DEL SCAN INICIAL ===
            if self.system_state == "SLEEP":
                interval = 300
            elif self.system_state == "IDLE":
                interval = self.segmentation_interval_idle
            else:  # ACTIVE
                interval = self.segmentation_interval_active

            frames_since_last = frame_count - self.last_segmentation_frame
            should_segment = frames_since_last >= interval

            if not should_segment:
                return self._use_cached_segmentation(frame, frame_count, objects_per_ruma, object_intersections)

        # === SI LLEGAMOS AQUÍ, SÍ SEGMENTAMOS ===
        self.last_segmentation_frame = frame_count
        
        # Determinar estado para logging
        if not self.initial_scan_complete:
            state_msg = "SCAN_INICIAL"
            interval = 5
        else:
            state_msg = self.system_state
            if self.system_state == "SLEEP":
                interval = 300
            elif self.system_state == "IDLE":
                interval = self.segmentation_interval_idle
            else:
                interval = self.segmentation_interval_active
        
        print(f"[RumaMonitor] Frame {frame_count}: Segmentando (modo {state_msg}, intervalo {interval})")
        
        # === CÓDIGO ORIGINAL DE SEGMENTACIÓN ===
        #result_seg = self.model_seg(frame, conf=0.5, verbose=False)
        result_seg = self.model_seg(
            frame, 
            conf=0.5, 
            verbose=False,
            imgsz=self.seg_imgsz  # NUEVO: usar tamaño correcto
        )
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

                        if not is_point_in_polygon(centroid, self.detection_zone):
                            continue

                        closest_ruma_id, distance = self.tracker.find_closest_ruma(centroid)

                        if closest_ruma_id is not None:
                            self.tracker.update_ruma(closest_ruma_id, mask, frame_count)
                            ruma = self.tracker.rumas[closest_ruma_id]

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
                            else:
                                display_percentage = ruma.percentage

                            if self.save_video:
                                label_text = f"R{ruma.id} | {display_percentage:.1f}%"
                                frame = put_text_with_background(
                                    frame, label_text, ruma.label_position,
                                    font_scale=0.6, color=self.TEXT_COLOR_WHITE
                                )

                        else:
                            self.tracker.add_candidate_ruma(mask, centroid, frame_count, frame.shape, self.transformer)

        self.tracker.clean_old_candidates(frame_count)
        
        # === GUARDAR EN CACHE ===
        self.cached_segmentation_data = {
            'interaction_alerts': interaction_alerts_to_send,
            'variation_alerts': variation_alerts_to_send,
            'frame': frame
        }

        return frame, interaction_alerts_to_send, variation_alerts_to_send

    def process_frame(self, frame, frame_count, fps):
        """Procesa un frame completo CON OPTIMIZACIONES"""
        
        t_start = time.time()
        
        if frame is None or frame.size == 0:
            print(f"[WARN] Frame {frame_count} inválido o vacío, saltando...")
            if hasattr(self, '_last_valid_frame_shape'):
                return np.zeros(self._last_valid_frame_shape, dtype=np.uint8)
            else:
                return np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        self._last_valid_frame_shape = frame.shape
        frame_with_drawings = frame.copy()

        # === DETECCIÓN (CON SKIP EN SLEEP) ===
        frame_with_drawings, movement_alerts, objects_per_ruma, object_intersections, has_activity = self.process_detections(
            frame_with_drawings, frame_count
        )
        t_detection = time.time()

        # === SEGMENTACIÓN (CONDICIONAL) ===
        frame_with_drawings, interaction_alerts, variation_alerts = self.process_segmentation(
            frame_with_drawings, frame_count, objects_per_ruma, object_intersections
        )
        t_segmentation = time.time()

        if self.save_video:
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
        
        t_drawing = time.time()

        # === ENVIAR ALERTAS ===
        for internal_id in movement_alerts:
            ruma_data = RumaInfo(
                id=None, percent=None, centroid=None, radius=None,
                centroid_homographic=None, radius_homographic=None
            )
            save_alert(
                alert_type='movimiento_zona', ruma_data=ruma_data,
                frame=frame_with_drawings, frame_count=frame_count, fps=fps,
                camera_sn=self.camera_sn, enterprise=self.enterprise,
                api_url=self.api_url, send=self.send, save=self.save,
                ruma_summary=self.tracker.ruma_summary,
                frame_shape=frame.shape, detection_zone=self.detection_zone
            )
        
        for (internal_id, ruma_id) in interaction_alerts:
            if ruma_id in self.tracker.rumas:
                ruma = self.tracker.rumas[ruma_id]
                ruma_data = RumaInfo(
                    id=ruma.id, percent=ruma.last_stable_percentage,
                    centroid=ruma.centroid, radius=ruma.radius,
                    centroid_homographic=ruma.centroid_homographic,
                    radius_homographic=ruma.radius_homographic
                )
                save_alert(
                    alert_type='interaccion_rumas', ruma_data=ruma_data,
                    frame=frame_with_drawings, frame_count=frame_count, fps=fps,
                    camera_sn=self.camera_sn, enterprise=self.enterprise,
                    api_url=self.api_url, send=self.send, save=self.save,
                    ruma_summary=self.tracker.ruma_summary,
                    frame_shape=frame.shape, detection_zone=self.detection_zone
                )
        
        for ruma_id in variation_alerts:
            if ruma_id in self.tracker.rumas:
                ruma = self.tracker.rumas[ruma_id]
                ruma_data = RumaInfo(
                    id=ruma.id, percent=ruma.percentage,
                    centroid=ruma.centroid, radius=ruma.radius,
                    centroid_homographic=ruma.centroid_homographic,
                    radius_homographic=ruma.radius_homographic
                )
                save_alert(
                    alert_type='variacion_rumas', ruma_data=ruma_data,
                    frame=frame_with_drawings, frame_count=frame_count, fps=fps,
                    camera_sn=self.camera_sn, enterprise=self.enterprise,
                    api_url=self.api_url, send=self.send, save=self.save,
                    ruma_summary=self.tracker.ruma_summary,
                    frame_shape=frame.shape, detection_zone=self.detection_zone
                )
        
        if self.tracker.new_ruma_created:
            ruma_id, frame_shape = self.tracker.new_ruma_created
            ruma = self.tracker.rumas[ruma_id]
            ruma_data = RumaInfo(
                id=ruma.id, percent=100.0,
                centroid=ruma.centroid, radius=ruma.radius,
                centroid_homographic=ruma.centroid_homographic,
                radius_homographic=ruma.radius_homographic
            )
            save_alert(
                alert_type='nueva_ruma', ruma_data=ruma_data,
                frame=frame_with_drawings, frame_count=frame_count, fps=fps,
                camera_sn=self.camera_sn, enterprise=self.enterprise,
                api_url=self.api_url, send=self.send, save=self.save,
                ruma_summary=self.tracker.ruma_summary,
                frame_shape=frame.shape, detection_zone=self.detection_zone
            )
            self.tracker.new_ruma_created = None
        
        t_alerts = time.time()
        self.object_tracker.cleanup_old_objects(frame_count)
        t_end = time.time()
        
        # === TIMERS ===
        if frame_count % 50 == 0:
            det_ms = (t_detection - t_start) * 1000
            seg_ms = (t_segmentation - t_detection) * 1000
            draw_ms = (t_drawing - t_segmentation) * 1000
            alerts_ms = (t_alerts - t_drawing) * 1000
            cleanup_ms = (t_end - t_alerts) * 1000
            total_ms = (t_end - t_start) * 1000
            
            print(f"\n[TIMING] Frame {frame_count} (Estado: {self.system_state})")
            print(f"  Detección:     {det_ms:6.1f}ms")
            print(f"  Segmentación:  {seg_ms:6.1f}ms")
            print(f"  Dibujos:       {draw_ms:6.1f}ms")
            print(f"  Alertas:       {alerts_ms:6.1f}ms")
            print(f"  Cleanup:       {cleanup_ms:6.1f}ms")
            print(f"  ─────────────────────────")
            print(f"  TOTAL:         {total_ms:6.1f}ms ({1000/total_ms:.1f} fps teórico)\n")

        return frame_with_drawings