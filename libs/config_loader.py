import yaml
import numpy as np


def load_camera_config(camera_number, config_path="mkdocs.yml"):

    print(f"Numero de cámara: {camera_number}", flush=True)

    # Leer archivo YAML
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    cameras = config.get("cameras", {})

    if camera_number not in cameras:
        raise ValueError(f"No hay configuración para la cámara {camera_number}")

    cam_config = cameras[camera_number]

    input_video = cam_config["input_video"]
    output_video = cam_config["output_video"]
    camera_sn = cam_config["camera_sn"]
    save_data = cam_config["save_data"]

    # Extraer el primer (y único) polígono definido para esta cámara
    polygon_list = cam_config["polygons"]
    if not polygon_list or not isinstance(polygon_list, list):
        raise ValueError(f"No se encontraron polígonos válidos para la cámara {camera_number}")

    detection_zone = np.array(polygon_list[0][1], dtype=np.int32)

    return input_video, output_video, detection_zone, camera_sn, save_data
