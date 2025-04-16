
import cv2
import yaml
import time
import numpy as np
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path



def debug_init(message):
    print("[DEBUG] ", message)

def draw_region_info(frame, polygons, counter, current_time):
    """
    Dibuja información de las regiones, incluyendo contadores
    """
    for i, polygon in polygons.items():
        # Dibujar polígono
        cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)    


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


# load de archivos
def path_yml(ruta_relativa: str) -> Path:
    """Args: ruta_relativa (ej: "file_ej/mkdocs_video.yml")""" 
    dir_actual = Path(__file__).parent
    full_path = dir_actual / ruta_relativa
    
    if full_path.exists():
        print("File Encontrado:", full_path)   
        return full_path 
    else: #Raises: FileNotFoundError: Si el archivo no existe
        raise FileNotFoundError(f" Archivo no found: {full_path}")