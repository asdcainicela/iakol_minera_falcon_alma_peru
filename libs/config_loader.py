import yaml
import numpy as np
import os
import cv2

from homography.homography_transformer import HomographyTransformer

def _read_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error al parsear YAML: {e}")

    return config

def _extract_camera_settings(config, camera_number):
    cameras = config.get("cameras", {})
    if camera_number not in cameras:
        raise ValueError(f"No hay configuración para la cámara {camera_number} en el archivo YAML.")

    cam_config = cameras[camera_number]

    try:
        input_video = cam_config["input_video"]
        output_video = cam_config["output_video"]
        camera_sn = cam_config["camera_sn"]
        save_data = cam_config["save_data"]
        input_pts = cam_config["input_homography"]
        output_pts = cam_config["output_homography"]
        polygon_list = cam_config.get("polygons")
        # save_video es opcional, por defecto False
        save_video = cam_config.get("save_video", False)
    except KeyError as e:
        raise ValueError(f"Falta una clave en la configuración de la cámara: {e}")

    if len(input_pts) < 4 or len(output_pts) < 4:
        raise ValueError("Se requieren al menos 4 puntos en input_homography y output_homography.")

    return input_video, output_video, camera_sn, save_data, input_pts, output_pts, polygon_list, save_video

def _build_transformer(input_pts, output_pts):
    try:
        return HomographyTransformer(input_pts=input_pts, output_pts=output_pts)
    except cv2.error as e:
        raise ValueError(f"Error al calcular la homografía: {e}")


def _extract_detection_zone(polygon_list, camera_number):
    if not polygon_list or not isinstance(polygon_list, list) or not polygon_list[0][1]:
        raise ValueError(f"No se encontró un polígono válido para la cámara {camera_number}")
    
    try:
        return np.array(polygon_list[0][1], dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Error al procesar el polígono: {e}")


# === FUNCIÓN PRINCIPAL  ===
def load_camera_config(camera_number, config_path="mkdocs.yml"):
    """
    Carga la configuración de una cámara específica desde un archivo YAML.

    Args:
        camera_number (int): Número de la cámara a cargar.
        config_path (str): Ruta al archivo de configuración YAML.

    Returns:
        Tuple[str, str, np.ndarray, str, str, HomographyTransformer, bool]:
            input_video, output_video, detection_zone, camera_sn, save_data, transformer, save_video
    """
    print(f"[INFO] Cargando configuración para cámara {camera_number} desde '{config_path}'", flush=True)

    config = _read_yaml_config(config_path)
    input_video, output_video, camera_sn, save_data, input_pts, output_pts, polygon_list, save_video = _extract_camera_settings(config, camera_number)
    transformer = _build_transformer(input_pts, output_pts)
    detection_zone = _extract_detection_zone(polygon_list, camera_number)

    return input_video, output_video, detection_zone, camera_sn, save_data, transformer, save_video