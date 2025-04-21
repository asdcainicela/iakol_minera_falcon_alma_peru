import os
import cv2
import yaml
import time
import requests
import argparse
import numpy as np
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
from collections import defaultdict


class PersonCounter:
    def __init__(self, polygons, config=None):
        """
        Inicializa el contador de personas
        """
        self.polygons = polygons
        self.api_url = "https://fn-va-panto.azurewebsites.net/api/camera-region-data"

        default_config = {
            'max_frames_missing': 720, #antes era 360
            'approaching_threshold': 500,
            'track_memory_time': 30.0,
            'min_entry_distance': 50,
            'debug': True
        }

        self.config = default_config
        if config is not None:
            self.config.update(config)

        # Estado principal de las personas
        self.person_states = defaultdict(lambda: {
            'region': None,
            'last_center': None,
            'frames_missing': 0,
            'entry_time': None,  # Cambiado: entry_time -> entry_frame
            'time_in_regions': defaultdict(float),
            'last_seen_time': None,  # Cambiado: last_seen_time -> last_seen_frame
            'velocity': (0, 0),
            'trajectory': [],
            'approaching_region': None,
            'distance_to_region': float('inf'),
            'original_id': None,
            'reassigned_from': None,
            'region_entries': set()  # Set para trackear regiones visitadas por este ID
        })

        # Historial persistente de tiempos por ID original
        self.persistent_times = defaultdict(lambda: defaultdict(float))
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.approaching_tracks = defaultdict(list)
        self.active_reassignments = set()
        self.active_original_id=set()
        self.current_frame_tracks = set()
        self.id_history = {}

        self.original_map = {}  # map: track_id -> canonical original_id

        # Tracking de entradas por ID original
        self.region_stats = defaultdict(lambda: {
            'current_count': 0,
            'total_entries': 0,
            'entries_time': defaultdict(int),  # Cambiado: entries_time -> entries_frame
            'total_time': defaultdict(float),
            'visited_by_original_ids': set()  # Set para trackear IDs originales que han visitado la región
        })


    def send_metadata(self, metadata):
        pass
        # print(f"Metadata a enviar: {metadata}")
        # try:
        #     response = requests.post(self.api_url, json=metadata)
        #     if response.status_code == 200:
        #         print("Metadata enviada con éxito.")
        #     else:
        #         print(f"Error al enviar metadata: {response.status_code}, {response.text}")
        # except Exception as e:
        #     print(f"Error al conectar con la API: {e}")

    def point_in_polygon(self, point, polygon):
        """Verifica si un punto está dentro de un polígono"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def get_region(self, point):
        """Obtiene la región en la que está un punto"""
        for i, polygon in self.polygons.items():
            if self.point_in_polygon(point, polygon):
                return i
        return None

    def distance_to_polygon(self, point, polygon):
        """Calcula la distancia mínima de un punto a un polígono"""
        return abs(cv2.pointPolygonTest(polygon, point, True))

    def calculate_center(self, box):
        """Calcula el centro de un bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_velocity(self, old_center, new_center, time_diff):
        """Calcula el vector de velocidad entre dos puntos"""
        if old_center is None or time_diff == 0:
            return (0, 0)
        return (
            (new_center[0] - old_center[0]) / time_diff,
            (new_center[1] - old_center[1]) / time_diff
        )

    def get_original_id(self, track_id):
        """
        Devuelve el canonical ID fijado en original_map.
        Si no existe entrada, devuelve el mismo track_id.
        """
        return self.original_map.get(track_id, track_id)

    def is_id_active(self, track_id, exclude_id=None):
        """
        Verifica si un ID está activo en cualquier parte de la imagen
        """
        for tid, state in self.person_states.items():
            if tid == exclude_id:
                continue

            # Verificar ID original
            if tid == track_id:
                if state['frames_missing'] == 0:
                    return True

            # Verificar ID reasignado
            if state.get('original_id') == track_id:
                if state['frames_missing'] == 0:
                    return True

        return False

    def update_approaching_tracks(self, track_id, center, current_time):
        """Actualiza la lista de tracks que se acercan a cada región"""
        min_distance = float('inf')
        closest_region = None

        for region_id, polygon in self.polygons.items():
            distance = self.distance_to_polygon(center, polygon)
            if distance < min_distance:
                min_distance = distance
                closest_region = region_id

        if min_distance < self.config['approaching_threshold']:
            self.approaching_tracks[closest_region] = [
                (tid, dist, time) for tid, dist, time in self.approaching_tracks[closest_region]
                if tid != track_id and
                current_time - time < self.config['track_memory_time'] and
                tid not in self.active_reassignments
            ]
            self.approaching_tracks[closest_region].append((track_id, min_distance, current_time))
            self.approaching_tracks[closest_region].sort(key=lambda x: x[1])

            # if self.config['debug']:
            #     print(f"Track {track_id} acercándose a región {closest_region}, distancia: {min_distance:.1f}")
 
    def find_existing_id(self, current_time, new_track_id, new_center):
        """
        Busca un ID existente que podría corresponder a esta nueva detección,
        descartando cualquier candidato que nunca haya existido y evitando IDs ya activos.
        """
        # 1) Sólo consideramos IDs que ya hayan estado en person_states o en id_history
        seen_ids = set(self.person_states.keys()) | set(self.id_history.keys())

        potential_tracks = []
        current_region = self.get_region(new_center)
        all_tracks = []

        for track_id, state in self.person_states.items():
            if track_id not in seen_ids:
                #if self.config['debug']:
                #    print(f"[IGNORADO] {track_id} nunca estuvo activo.")
                continue
            if track_id == new_track_id:
                continue

            orig = self.get_original_id(track_id)
            if orig == new_track_id:
                continue

            # Sólo candidatos que estén “perdidos” (frames_missing > 0)
            if state['frames_missing'] == 0:
                continue

            # No reasignar a un original que ya estamos usando este frame
            if orig in self.active_original_id:
                continue

            # Calcular distancia real y predicha
            if state['last_center'] is not None:
                dist = np.linalg.norm(np.array(new_center) - np.array(state['last_center']))
                if state['velocity'] != (0, 0):
                    dt = current_time - state['last_seen_time']
                    pred = np.array(state['last_center']) + np.array(state['velocity'])*dt
                    dist = min(dist, np.linalg.norm(np.array(new_center)-pred))

                region_match = 0.5 if state['region'] == current_region else 1.0
                info = {
                    'current_id': track_id,
                    'original_id': orig,
                    'distance': dist * region_match,
                    'real_distance': dist,
                    'last_seen_time': state['last_seen_time'],
                    'frames_missing': state['frames_missing'],
                    'region': state['region']
                }
                all_tracks.append(info)
                if state['frames_missing'] <= self.config['max_frames_missing']*2:
                    potential_tracks.append(info)

        # 2) Primero intentamos con potential_tracks
        if potential_tracks:
            potential_tracks.sort(key=lambda x: (
                x['frames_missing'], -x['last_seen_time'], x['distance']
            ))
            max_dist = self.config['approaching_threshold'] * 2
            for t in potential_tracks:
                if t['real_distance'] > (max_dist * (2 if t['region']==current_region else 1)):
                    #if self.config['debug']:
                    #    print(f"[DESCARTADO] {t['original_id']} dist{t['real_distance']:.1f} > umbral")
                    continue
                if not self.is_id_active(t['original_id'], exclude_id=new_track_id):
                    #if self.config['debug']:
                    #    print(f"[REASIGNANDO] {t['original_id']} para {new_track_id}")
                    self.person_states[t['original_id']]['frames_missing'] = 0
                    return t['original_id']

        # 3) Si no hay ninguno válido, no reasignamos
        #if self.config['debug']:
        #    print(f"[SIN REASIGNACIÓN] No hay candidato válido para {new_track_id}")
        return None

    def handle_transition(self, track_id, old_region, new_region, current_time, camera_id):
        """
        Maneja las transiciones entre regiones
        """
        if old_region != new_region:
            self.transition_counts[old_region][new_region] += 1
            original_id = self.get_original_id(track_id)

            # Actualizar contadores de región
            if new_region is not None:
                # Solo contar como nueva entrada si es la primera vez para este ID original
                if original_id not in self.region_stats[new_region]['visited_by_original_ids']:
                    self.region_stats[new_region]['total_entries'] += 1
                    self.region_stats[new_region]['visited_by_original_ids'].add(original_id)

                # Actualizar frame de última entrada
                self.region_stats[new_region]['entries_time'][original_id] = current_time

            # Actualizar tiempos
            if old_region is not None and self.person_states[track_id]['entry_time'] is not None:
                time_spent = current_time - self.person_states[track_id]['entry_time']
                self.person_states[track_id]['time_in_regions'][old_region] += time_spent
                self.persistent_times[original_id][old_region] += time_spent
                self.region_stats[old_region]['total_time'][original_id] += time_spent

            entry_time = self.person_states[track_id]['entry_time']
            self.person_states[track_id]['entry_time'] = current_time
            self.person_states[original_id]['entry_time'] = current_time
            self.person_states[original_id]['region'] = new_region
            self.person_states[original_id]['frames_missing'] = 0

            # Registrar la región en el set de regiones visitadas
            self.person_states[track_id]['region_entries'].add(new_region)

            # Dando formato de tiempo
            entry_format = datetime.fromtimestamp(entry_time).strftime("%Y-%m-%d %H:%M:%S")
            current_format = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")

            # Generar metadata según los casos especificados
            if old_region is None and new_region is not None:
                metadata = {
                    "id_region": new_region,
                    "deviceSn": camera_id,
                    "in": True,
                    "out": False,
                    "region_time": 0,
                    "entry_time": current_format,
                    "out_time": current_format
                }
                self.send_metadata(metadata)

            elif old_region is not None and new_region is None:
                metadata = {
                    "id_region": old_region,
                    "deviceSn": camera_id,
                    "in": False,
                    "out": True,
                    "region_time": current_time - entry_time,
                    "entry_time": entry_format,
                    "out_time": current_format
                }
                self.send_metadata(metadata)

            elif old_region is not None and new_region is not None:
                metadata_new = {
                    "id_region": new_region,
                    "deviceSn": camera_id,
                    "in": True,
                    "out": False,
                    "region_time": 0,
                    "entry_time": current_format,
                    "out_time": current_format
                }
                metadata_old = {
                    "id_region": old_region,
                    "deviceSn": camera_id,
                    "in": False,
                    "out": True,
                    "region_time": current_time - entry_time,
                    "entry_time": entry_format,
                    "out_time": current_format
                }
                self.send_metadata(metadata_new)
                self.send_metadata(metadata_old)

            #if self.config['debug']:
                #print(f"Transición: track {track_id} (original: {original_id}) de región {old_region} a {new_region}")
                #if new_region is not None:
                #    print(f"  Tiempo total en región {new_region}: {self.get_total_time_in_region(track_id, new_region, current_time):.1f}s")

    def process_frame(self, results, current_time, camera_id):
        """Procesa un frame y actualiza el estado del contador"""
        all_current_tracks = set()
        new_detections = []

        # 1) Revisión de cada detección en el frame
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if not hasattr(box, 'id') or box.id is None:
                    continue

                # ————— Paso 2: inicializar mapping canonical —————
                raw_id = int(box.id.item())
                if raw_id not in self.original_map:
                    # la primera vez que vemos raw_id, su original es él mismo
                    self.original_map[raw_id] = raw_id
            
                #raw_id = int(box.id.item())
                xyxy = box.xyxy[0].cpu().numpy()
                center = self.calculate_center(xyxy)
                current_region = self.get_region(center)

                # 2) REASIGNACIÓN GLOBAL: aunque esté fuera de región,
                #    intentamos mapear raw_id → existing_id
                existing_id = self.find_existing_id(current_time, raw_id, center)
                if existing_id is not None and existing_id != raw_id:
                    #if self.config['debug']:
                    #    prev_reg = self.person_states[existing_id]['region']
                    #    print(f"[GLOBAL REASIGNACIÓN] {raw_id} → {existing_id} (antes en R{prev_reg})")
                    # reset de frames missing y marcar reasignación
                    # ————— Paso 3: heredar el canonical de existing_id —————
                    canon = self.original_map.get(existing_id, existing_id)
                    self.original_map[raw_id] = canon
                    #if self.config['debug']:
                    #    print(f"[MAP CANÓNICO] raw_id {raw_id} mapeado a canonical {canon}")

                    # reset frames, logging habitual…
                    self.person_states[existing_id]['frames_missing'] = 0
                    self.active_reassignments.add(existing_id)
                    self.id_history[raw_id] = {
                        'original_id': existing_id,
                        'reassigned_at_frame': current_time,
                        'region': current_region
                    }
                    track_id = existing_id
                else:
                    track_id = raw_id

                # 3) Si está fuera de todas las regiones
                if current_region is None:
                    self.update_approaching_tracks(track_id, center, current_time)
                    new_detections.append({
                        'track_id': track_id,
                        'center': center,
                        'region': None,
                        'box': xyxy
                    })

                # 4) Si está dentro de una región válida
                else:
                    # Track nuevo dentro de región
                    if track_id not in self.person_states:
                        #if self.config['debug']:
                        #    print(f"[CREACIÓN] Nuevo ID {track_id} en R{current_region}")
                        # Intentar reasignar dentro de región (llama de nuevo por si alguien entró directamente)
                        existing2 = self.find_existing_id(current_time, track_id, center)
                        if existing2 is not None and existing2 != track_id:
                            #if self.config['debug']:
                            #    print(f"[REASIGNACIÓN] {track_id} → {existing2}")
                            self.person_states[existing2]['frames_missing'] = 0
                            self.active_reassignments.add(existing2)
                            self.id_history[track_id] = {
                                'original_id': existing2,
                                'reassigned_at_frame': current_time,
                                'region': current_region
                            }
                            # copiar estado al nuevo track
                            self.person_states[track_id] = self.person_states[existing2].copy()
                            self.person_states[track_id].update({
                                'original_id': existing2,
                                'reassigned_from': track_id,
                                'last_center': center,
                                'frames_missing': 0,
                                'last_seen_time': current_time,
                                'entry_time': self.person_states[existing2]['entry_time'],
                                'region': current_region,
                            })

                    new_detections.append({
                        'track_id': track_id,
                        'center': center,
                        'region': current_region,
                        'box': xyxy
                    })

                # 5) Marcar que vimos este track en el frame actual
                all_current_tracks.add(track_id)
                self.current_frame_tracks.add(track_id)

        # 6) Evitar duplicados: si dos raw_ids comparten un mismo original_id,
        #    anula el original_id del segundo para desambiguar
        for tid in self.current_frame_tracks:
            orig = self.person_states.get(tid, {}).get('original_id')
            if orig is not None and orig in self.current_frame_tracks and orig != tid:
                #if self.config['debug']:
                #    print(f"[DUPLICADO] {tid} y {orig} activos juntos: anulando original de {tid}")
                self.person_states[tid]['original_id'] = None

        self.current_frame_tracks.clear()

        # 7) Limpiar reasignaciones inactivas
        self.active_reassignments = {
            tid for tid in self.active_reassignments
            if any(st['original_id'] == tid for st in self.person_states.values())
        }

        # 8) Procesar cada nueva detección y manejar transiciones
        for det in new_detections:
            tid = det['track_id']
            ctr = det['center']
            reg = det['region']

            prev = self.person_states[tid]
            prev_reg = prev['region']
            prev_time = prev['last_seen_time']

            # Velocidad
            if prev_time is not None:
                dt = current_time - prev_time
                vel = self.calculate_velocity(prev['last_center'], ctr, dt)
            else:
                vel = (0, 0)

            # Actualizar estado
            self.person_states[tid].update({
                'region': reg,
                'last_center': ctr,
                'frames_missing': 0,
                'last_seen_time': current_time,
                'velocity': vel,
                'trajectory': prev['trajectory'][-9:] + [ctr] if prev['trajectory'] else [ctr]
            })

            # Inicializar entry_time
            if prev['entry_time'] is None:
                self.person_states[tid]['entry_time'] = current_time

            # Transición de regiones
            if prev_reg != reg:
                self.handle_transition(tid, prev_reg, reg, current_time, camera_id)

        # 9) Actualizar tracks perdidos (sin borrarlos)
        for tid in list(self.person_states.keys()):
            if tid not in all_current_tracks:
                st = self.person_states[tid]
                st['frames_missing'] += 1
                if st['frames_missing'] > self.config['max_frames_missing']:
                    #if self.config['debug']:
                    #    print(f"[RETENIDO] Track {tid} timeout; sigue en memoria with frames_missing={st['frames_missing']}")
                    if st['region'] is not None and st['entry_time'] is not None:
                        spent = current_time - st['entry_time']
                        st['time_in_regions'][st['region']] += spent
                        orig = self.get_original_id(tid)
                        self.persistent_times[orig][st['region']] += spent
                    # ya no hacemos del self.person_states[tid]

        return self.transition_counts, self.get_average_times(), self.id_history

    def get_average_times(self):
        """Calcula el tiempo promedio de permanencia en cada región"""
        total_times = defaultdict(float)
        count_persons = defaultdict(int)

        # Procesar tiempos de tracks activos
        for track_id, person_data in self.person_states.items():
            original_id = self.get_original_id(track_id)

            # Combinar tiempos actuales con persistentes
            combined_times = defaultdict(float)
            for region, time_spent in person_data['time_in_regions'].items():
                combined_times[region] = time_spent + self.persistent_times[original_id][region]

            for region, time_spent in combined_times.items():
                if time_spent > 0:
                    total_times[region] += time_spent
                    count_persons[region] += 1

        # Calcular promedios
        avg_times = {}
        for region in total_times:
            if count_persons[region] > 0:
                avg_times[region] = total_times[region] / count_persons[region]
            else:
                avg_times[region] = 0

        return avg_times

    def get_persistent_times(self, track_id, current_time):
        """
        Obtiene los tiempos acumulados totales para un ID específico
        incluyendo tiempos actuales y persistentes
        """
        original_id = self.get_original_id(track_id)
        total_times = defaultdict(float)

        # Obtener tiempos persistentes
        for region, time in self.persistent_times[original_id].items():
            total_times[region] += time

        # Agregar tiempos actuales si el track está activo
        if track_id in self.person_states:
            state = self.person_states[track_id]

            # Añadir tiempo acumulado en región actual
            if state['region'] is not None and state['entry_time'] is not None:
                time_in_current = current_time - state['entry_time']
                total_times[state['region']] += time_in_current

            # Añadir tiempos acumulados en otras regiones
            for region, time_spent in state['time_in_regions'].items():
                total_times[region] += time_spent

        return dict(total_times)

    def get_total_time_in_region(self, track_id, region, current_time):
        """
        Obtiene el tiempo total acumulado en una región para un ID
        """
        total_time = 0
        state = self.person_states[track_id]
        current_time_in_region = current_time - state['entry_time']
        total_time += current_time_in_region

        return total_time

    def get_region_stats(self, region_id, current_time):
        """
        Obtiene estadísticas actualizadas de una región
        """
        stats = self.region_stats[region_id]

        # Contar personas actualmente en la región
        current_count = 0
        total_time = defaultdict(float)

        for track_id, state in self.person_states.items():
            if state['region'] == region_id:
                original_id = self.get_original_id(track_id)
                if self.is_id_active(track_id):
                    current_count += 1

                # Actualizar tiempo total
                total_time[original_id] = self.get_total_time_in_region(track_id, region_id, current_time)

        # Calcular tiempo promedio
        avg_time = 0
        if total_time:
            avg_time = sum(total_time.values()) / len(total_time)

        return {
            'current_count': current_count,
            'total_entries': stats['total_entries'],  # Ahora refleja entradas únicas por ID original
            'avg_time': avg_time,
            'unique_visitors': len(stats['visited_by_original_ids'])  # Número de visitantes únicos
        }

    def print_track_info(self, track_id, current_time):
        """
        Imprime información detallada sobre un track específico
        """
        if track_id in self.person_states:
            state = self.person_states[track_id]
            original_id = self.get_original_id(track_id)

            print(f"\nInformación del Track {track_id}:")
            print(f"ID Original: {original_id}")
            print(f"Región actual: {state['region']}")
            print(f"Frames perdidos: {state['frames_missing']}")
            print("\nTiempos acumulados por región:")

            total_times = self.get_persistent_times(track_id, current_time)
            for region, time_spent in total_times.items():
                print(f"  Región {region}: {time_spent:.1f} segundos")

            if track_id in self.id_history:
                history = self.id_history[track_id]
                print(f"\nHistorial de reasignación:")
                print(f"  Reasignado de: ID {history['original_id']}")
                print(f"  Frame de reasignación: {history['reassigned_at_frame']}")
        else:
            print(f"El track {track_id} no está activo actualmente")

            
# Funciones auxiliares para visualización
def draw_text_with_background(frame, text, pos, font_scale=1, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5):
    """
    Dibuja texto con un fondo semi-transparente para mejorar la legibilidad
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = pos
    padding = 5
    bg_rect = ((x - padding, y - text_height - padding),
               (x + text_width + padding, y + padding))

    overlay = frame.copy()
    cv2.rectangle(overlay, bg_rect[0], bg_rect[1], bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def draw_tracking_info(frame, box, track_id, region, confidence, person_state, counter, current_time):
    """
    Dibuja la información de tracking sobre cada persona detectada
    """
    x1, y1, x2, y2 = map(int, box)

    # Dibujar bounding box
    original_id = counter.get_original_id(track_id)
    if original_id != track_id:
        # Naranja para IDs reasignados
        box_color = (0, 165, 255)
    else:
        # Verde para IDs originales
        box_color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Preparar texto con ID y reasignación
    if original_id != track_id:
        text = f"ID:{track_id} -> ID:{original_id}"  # Formato corregido
    else:
        text = f"ID:{track_id}"

    if region is not None:
        text += f" | R{region}"
    text += f" | {confidence:.2f}"

    # Agregar tiempo en región si está disponible
    if region is not None and track_id in counter.person_states:
        total_time = counter.get_total_time_in_region(track_id, region, current_time)
        if total_time > 0:
            text += f" | {total_time:.1f}s"

    # Dibujar texto con fondo
    draw_text_with_background(
        frame,
        text,
        (x1, y1 - 10),
        font_scale=0.4,
        thickness=1,
        text_color=(255, 255, 255),
        bg_color=(0, 100, 0) if original_id == track_id else (165, 100, 0)
    )

def draw_region_info(frame, polygons, counter, current_time):
    """
    Dibuja información de las regiones, incluyendo contadores
    """
    for i, polygon in polygons.items():
        # Dibujar polígono
        cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

def time_to_frames(time_str, fps):
    """Convierte tiempo en formato mm:ss a número de frames"""
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return int(total_seconds * fps)

# Cargar configuración desde el archivo .yml
def load_camera_config(camera_number, config_path="mkdocs_video.yml"):
    print(f"Numero de camara: {camera_number}", flush=True)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    cameras = config.get("cameras", {})
    # print(f"Camaras: {cameras}", flush=True)
    if camera_number not in cameras:
        raise ValueError(f"No hay configuración para la cámara {camera_number}")

    cam_config = cameras[camera_number]
    input_video = cam_config["input_video"]
    output_video = cam_config["output_video"]
    camera_sn = cam_config["camera_sn"]
    # polygons = [np.array(polygon, np.int32) for polygon in cam_config["polygons"]]
    polygons = {polygon[0]: np.array(polygon[1], np.int32) for polygon in cam_config["polygons"]}

    return input_video, output_video, polygons, camera_sn

def process_video(input_path, output_path, model_path, polygons, camera_id, show_display=False, show_drawings=False, save_frame=False):
    """
    Procesa un video con el contador de personas
    """
    cap = cv2.VideoCapture(input_path)
    model = YOLO(model_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Verificar si la conexión fue exitosa
    if not cap.isOpened():
        print("Error al conectar con el stream RTSP")
        return
    
    print("Conexión establecida. Mostrando stream...")

    # Calcular los frames de inicio y fin
    start_time_sec = 25 * 60 + 30    # 25 minutos con 30 segundos
    end_time_sec = 28 * 60   # 28 minutos con 30 segundos
    start_frame = int(start_time_sec * stream_fps)
    end_frame = int(end_time_sec * stream_fps)

    # Mover el puntero al frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    if save_frame:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, stream_fps, (width, height))

    counter = PersonCounter(polygons)

    while cap.isOpened() and current_frame <= end_frame:

        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Error al recibir frame. Saliendo...")
            break

        results = model.track(frame, persist=True, classes=[0], conf=0.5, verbose=False, half=True)

        # Procesar resultados
        transitions, avg_times, id_history = counter.process_frame(results, start_time, camera_id)

        if show_drawings:
            # Dibujar información de regiones y tracking
            draw_region_info(frame, polygons, counter, start_time)

            # Dibujar información de tracking para cada persona
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if not hasattr(box, 'id'):
                        continue
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf.item())
                    region = counter.get_region(counter.calculate_center(xyxy))
                    person_state = counter.person_states.get(track_id, {})

                    draw_tracking_info(frame, xyxy, track_id, region, conf, person_state, counter, start_time)

        if save_frame:
            out.write(frame)

        if show_display:
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("RTSP Stream", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)

        current_frame += 1
        end_time = time.time()
        fps = 1/(end_time-start_time)
        # print(f"FPS del procesamiento: {fps:.2f}, {len(results[0].boxes)} personas en el frame", flush=True)

    cap.release()
    if save_frame:
        out.release()
    if show_display:
        cv2.destroyAllWindows()

    return transitions, avg_times, id_history

# Ejemplo de uso del código
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar video desde una cámara específica")
    parser.add_argument("camera_number", type=int, help="Número de cámara a usar (1-7)")
    parser.add_argument("--config", type=str, default="mkdocs_video.yml", help="Ruta al archivo de configuración YAML")
    args = parser.parse_args()

    try:
        input_video, output_video, polygons, camera_sn= load_camera_config(args.camera_number, args.config)
    except ValueError as e:
        print(e)
        exit(1)

    model_path = "yolo11n.engine"

    # Procesar video con las opciones especificadas
    transitions, avg_times, id_history = process_video(
        input_path=input_video,
        output_path=output_video,
        model_path=model_path,
        polygons=polygons,
        camera_id=camera_sn,
        show_display=False,
        show_drawings=True,
        save_frame=True
    )

    # Imprimir resultados finales
    print("\n=== Resultados Finales ===")
    print("\nTransiciones entre regiones:")
    for from_region in sorted(transitions.keys(), key=lambda x: str(x)):
        for to_region in sorted(transitions[from_region].keys(), key=lambda x: str(x)):
            count = transitions[from_region][to_region]
            print(f"De {from_region if from_region is not None else 'fuera':<6} "
                  f"a {to_region if to_region is not None else 'fuera':<6}: {count:>3}")

    print("\nTiempos promedio de permanencia por región:")
    for region in sorted(avg_times.keys()):
        print(f"Región {region}: {avg_times[region]:.1f} segundos")

    print("\nHistorial de reasignaciones de IDs:")
    for track_id, history in sorted(id_history.items()):
        print(f"ID {track_id} -> ID original {history['original_id']}")
