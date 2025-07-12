import yaml
import numpy as np

def load_camera_config(camera_number, config_path="mkdocs.yml"):
    """
    Carga la configuración de una cámara específica desde un archivo YAML.

    Args:
        camera_number (int): Número de la cámara a cargar.
        config_path (str): Ruta al archivo de configuración YAML.

    Returns:
        Tuple[str, str, np.ndarray, str, str]: input_video, output_video, detection_zone, camera_sn, save_data
    """
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

    # Obtener el primer polígono y convertirlo a np.ndarray
    polygon_list = cam_config.get("polygons")
    if not polygon_list or not isinstance(polygon_list, list):
        raise ValueError(f"No se encontraron polígonos válidos para la cámara {camera_number}")

    detection_zone = np.array(polygon_list[0][1], dtype=np.int32)

    return input_video, output_video, detection_zone, camera_sn, save_data
