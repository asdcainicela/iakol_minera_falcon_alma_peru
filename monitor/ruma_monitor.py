"""
monitor/ruma_monitor.py (VERSIÓN CORREGIDA)

FIX CRÍTICO: El scan inicial debe ser PRIORITARIO sobre cualquier optimización.
"""

import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

from utils.geometry import is_point_in_polygon, calculate_intersection
from alerts.alert_manager import save_alert
from utils.draw import put_text_with_background, draw_zone_and_status
from monitor.ruma_tracker import RumaTracker
from monitor.object_tracker import ObjectTracker
from alerts.alert_info import RumaInfo


class RumaMonitor:
    def __init__(self, model_det_path, model_seg_path, detection_zone, camera_sn, 
                 api_url, transformer, save_video=False,
                 segmentation_interval_idle=100,
                 segmentation_interval_active=10,
                 activity_cooldown_frames=200,
                 detection_skip_idle=3):
        """
        Monitor CON THREADING optimizado.
        
        FIX: Scan inicial FORZADO - ignora todos los intervalos durante los primeros 150 frames
        """
        self.api_url = api_url 
        
        # Detectar TensorRT
        self.is_tensorrt_det = model_det_path.endswith('.engine')
        self.is_tensorrt_seg = model_seg_path.endswith('.engine')
        
        self.model_det = YOLO(model_det_path)
        self.model_seg = YOLO(model_seg_path)
        
        self.det_imgsz = 1024 if self.is_tensorrt_det else 640
        self.seg_imgsz = 1024 if self.is_tensorrt_seg else 640
        
        print(f"[RumaMonitor] Modelos:")
        print(f"  Det: {'TensorRT' if self.is_tensorrt_det else 'PyTorch'} @ {self.det_imgsz}px")
        print(f"  Seg: {'TensorRT' if self.is_tensorrt_seg else 'PyTorch'} @ {self.seg_imgsz}px")

        self.detection_zone = detection_zone
        self.camera_sn = camera_sn
        self.enterprise = 'alma'
        self.save_video = save_video

        self.tracker = RumaTracker()
        self.object_tracker = ObjectTracker(
            interaction_threshold=40,
            max_distance_match=150
        )

        # Colores
        self.RUMA_COLOR = (0, 255, 0)
        self.PERSON_COLOR = (255, 255, 0)
        self.VEHICLE_COLOR = (0, 0, 255)
        self.TEXT_COLOR_WHITE = (255, 255, 255)
        self.TEXT_COLOR_GREEN = (0, 255, 0)
        self.TEXT_COLOR_RED = (0, 0, 255)

        self.transformer = transformer
        self.send = True
        self.save = False
        
        # === FIX: SCAN INICIAL MÁS LARGO Y PRIORITARIO ===
        self.INITIAL_SCAN_FRAMES = 150  # Aumentado de 50 a 150
        self.INITIAL_SCAN_INTERVAL = 1  # CADA frame (no cada 2)
        
        self.segmentation_interval_idle = segmentation_interval_idle
        self.segmentation_interval_active = segmentation_interval_active
        self.activity_cooldown_frames = activity_cooldown_frames
        self.detection_skip_idle = detection_skip_idle
        
        # Estado
        self.initial_scan_complete = False
        self.system_state = "INITIAL_SCAN"  # Nuevo estado prioritario
        self.frames_without_activity = 0
        self.last_segmentation_frame = -999
        
        print(f"[RumaMonitor] Configuración:")
        print(f"  🔍 SCAN INICIAL: {self.INITIAL_SCAN_FRAMES} frames, cada {self.INITIAL_SCAN_INTERVAL} frame")
        print(f"  ⏸️  Seg IDLE: cada {segmentation_interval_idle} frames")
        print(f"  ▶️  Seg ACTIVE: cada {segmentation_interval_active} frames")
        print(f"  ⏱️  Cooldown: {activity_cooldown_frames} frames\n")

    def process_detections(self, frame, frame_count):
        """Detección de personas/vehículos (sin cambios)"""
        movement_alerts = set()
        objects_per_ruma = {}
        object_intersections = {}
        has_activity_in_zone = False

        # Durante scan inicial, no saltar detecciones
        if not self.initial_scan_complete:
            should_detect = True
        elif self.system_state == "SLEEP":
            should_detect = frame_count % self.detection_skip_idle == 0
        else:
            should_detect = True

        if not should_detect:
            return frame, movement_alerts, objects_per_ruma, object_intersections, has_activity_in_zone

        result_det = self.model_det.predict(
            frame, 
            conf=0.5,
            verbose=False,
            imgsz=self.det_imgsz
        )

        if result_det and len(result_det) > 0:
            boxes = result_det[0].boxes

            if boxes and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > 0.5:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        centroid = (center_x, center_y)
                        
                        object_type = 'person' if cls == 0 else 'vehicle'
                        in_polygon = is_point_in_polygon(centroid, self.detection_zone)
                        
                        internal_id = self.object_tracker.update_or_create_object(
                            yolo_track_id=None,
                            centroid=centroid,
                            bbox=(x1, y1, x2, y2),
                            object_type=object_type,
                            frame_count=frame_count,
                            in_polygon=in_polygon
                        )
                        
                        if in_polygon:
                            has_activity_in_zone = True
                        
                        if self.object_tracker.check_movement_alert(internal_id):
                            movement_alerts.add(internal_id)
                        
                        if self.save_video:
                            color = self.PERSON_COLOR if cls == 0 else self.VEHICLE_COLOR
                            label = f'{object_type} ID:{internal_id} [{conf:.2f}]'
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            frame = put_text_with_background(
                                frame, label, (x1, y1 - 5),
                                color=self.TEXT_COLOR_WHITE, font_scale=0.6
                            )
                        
                        # Interacción con rumas
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

        # === FIX: ACTUALIZAR ESTADO SOLO DESPUÉS DEL SCAN ===
        if self.initial_scan_complete:
            if has_activity_in_zone:
                self.frames_without_activity = 0
                if self.system_state != "ACTIVE":
                    print(f"[Monitor] Frame {frame_count}: TRANSICIÓN {self.system_state} → ACTIVE")
                    self.system_state = "ACTIVE"
            else:
                self.frames_without_activity += 1
                
                if self.system_state == "ACTIVE" and self.frames_without_activity >= self.activity_cooldown_frames:
                    print(f"[Monitor] Frame {frame_count}: TRANSICIÓN ACTIVE → IDLE")
                    self.system_state = "IDLE"
                
                elif self.system_state == "IDLE" and self.frames_without_activity >= 600:
                    print(f"[Monitor] Frame {frame_count}: TRANSICIÓN IDLE → SLEEP")
                    self.system_state = "SLEEP"

        return frame, movement_alerts, objects_per_ruma, object_intersections, has_activity_in_zone

    def process_segmentation(self, frame, frame_count, objects_per_ruma, object_intersections):
        """
        FIX CRÍTICO: Scan inicial PRIORITARIO sobre todo.
        """
        
        # === FASE 1: SCAN INICIAL (PRIORIDAD MÁXIMA) ===
        if not self.initial_scan_complete:
            if frame_count <= self.INITIAL_SCAN_FRAMES:
                # Durante scan inicial: segmentar cada N frames
                frames_since_last = frame_count - self.last_segmentation_frame
                
                if frames_since_last >= self.INITIAL_SCAN_INTERVAL:
                    print(f"[Monitor] 🔍 SCAN INICIAL: Frame {frame_count}/{self.INITIAL_SCAN_FRAMES}")
                    self.last_segmentation_frame = frame_count
                else:
                    # Skip - reusar última segmentación si existe
                    return self._reuse_last_segmentation(frame, objects_per_ruma, object_intersections)
            else:
                # SCAN COMPLETADO
                self.initial_scan_complete = True
                self.system_state = "IDLE"  # Cambiar a IDLE después del scan
                print(f"[Monitor] ✅ SCAN INICIAL COMPLETADO")
                print(f"[Monitor]    Rumas detectadas: {len(self.tracker.rumas)}")
                print(f"[Monitor]    Pasando a modo IDLE\n")
                return frame, set(), set()
        
        # === FASE 2: MODO NORMAL (DESPUÉS DEL SCAN) ===
        else:
            # Determinar intervalo según estado
            if self.system_state == "SLEEP":
                interval = 300
            elif self.system_state == "IDLE":
                interval = self.segmentation_interval_idle
            else:  # ACTIVE
                interval = self.segmentation_interval_active

            frames_since_last = frame_count - self.last_segmentation_frame
            
            if frames_since_last < interval:
                return self._reuse_last_segmentation(frame, objects_per_ruma, object_intersections)
            
            self.last_segmentation_frame = frame_count
            print(f"[Monitor] Segmentando (modo {self.system_state}, intervalo {interval}) - Frame {frame_count}")

        # === EJECUTAR SEGMENTACIÓN ===
        result_seg = self.model_seg(
            frame, 
            conf=0.5, 
            verbose=False,
            imgsz=self.seg_imgsz
        )
        
        interaction_alerts = set()
        variation_alerts = set()
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
                                    variation_alerts.add(closest_ruma_id)
                            
                            ruma.was_interacting = is_interacting

                            if is_interacting:
                                ruma.frames_without_interaction = 0
                                ruma.last_stable_percentage = ruma.percentage
                                
                                for internal_id in objects_per_ruma[closest_ruma_id]:
                                    intersecting = object_intersections.get(internal_id, set())
                                    should_alert, confirmed_ruma = self.object_tracker.update_interaction(
                                        internal_id, intersecting
                                    )
                                    if should_alert:
                                        interaction_alerts.add((internal_id, confirmed_ruma))
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
                            # Nueva ruma
                            self.tracker.add_candidate_ruma(mask, centroid, frame_count, frame.shape, self.transformer)

        self.tracker.clean_old_candidates(frame_count)

        return frame, interaction_alerts, variation_alerts

    def _reuse_last_segmentation(self, frame, objects_per_ruma, object_intersections):
        """Reutiliza última segmentación cuando se hace skip"""
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
        
        # Procesar interacciones con la última segmentación conocida
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

    def process_frame(self, frame, frame_count, fps):
        """Procesa frame (sin cambios en la lógica de alertas)"""
        if frame is None or frame.size == 0:
            print(f"[WARN] Frame {frame_count} inválido")
            if hasattr(self, '_last_valid_frame_shape'):
                return np.zeros(self._last_valid_frame_shape, dtype=np.uint8)
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        self._last_valid_frame_shape = frame.shape
        frame_with_drawings = frame.copy()

        frame_with_drawings, movement_alerts, objects_per_ruma, object_intersections, _ = self.process_detections(
            frame_with_drawings, frame_count
        )

        frame_with_drawings, interaction_alerts, variation_alerts = self.process_segmentation(
            frame_with_drawings, frame_count, objects_per_ruma, object_intersections
        )

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

        # Enviar alertas (código original sin cambios)
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
        
        self.object_tracker.cleanup_old_objects(frame_count)

        return frame_with_drawings