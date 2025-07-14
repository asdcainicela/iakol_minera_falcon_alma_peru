import yaml
import numpy as np
import os
import cv2

from homography.homography_transformer import HomographyTransformer

def load_camera_config(camera_number, config_path="mkdocs.yml"):
    """
    Carga la configuración de una cámara específica desde un archivo YAML.

    Args:
        camera_number (int): Número de la cámara a cargar.
        config_path (str): Ruta al archivo de configuración YAML.

    Returns:
        Tuple[str, str, np.ndarray, str, str, HomographyTransformer]:
            input_video, output_video, detection_zone, camera_sn, save_data, transformer

    Raises:
        FileNotFoundError: Si el archivo YAML no existe.
        yaml.YAMLError: Si hay un error de sintaxis en el YAML.
        ValueError: Si faltan claves o hay datos inválidos.
        cv2.error: Si hay un error al calcular la homografía.
    """
    print(f"[INFO] Cargando configuración para cámara {camera_number} desde '{config_path}'", flush=True)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error al parsear YAML: {e}")

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
    except KeyError as e:
        raise ValueError(f"Falta una clave en la configuración de la cámara: {e}")

    if len(input_pts) < 4 or len(output_pts) < 4:
        raise ValueError("Se requieren al menos 4 puntos en input_homography y output_homography.")

    try:
        transformer = HomographyTransformer(input_pts=input_pts, output_pts=output_pts)
    except cv2.error as e:
        raise ValueError(f"Error al calcular la homografía: {e}")

    polygon_list = cam_config.get("polygons")
    if not polygon_list or not isinstance(polygon_list, list) or not polygon_list[0][1]:
        raise ValueError(f"No se encontró un polígono válido para la cámara {camera_number}")

    try:
        detection_zone = np.array(polygon_list[0][1], dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Error al procesar el polígono: {e}")

    return input_video, output_video, detection_zone, camera_sn, save_data, transformer
