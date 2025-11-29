import requests
from datetime import datetime

from alerts.alert_info import RumaInfo, AlertContext
from utils.convert_b64 import frame_to_base64, save_b64_image

def send_metadata(metadata: dict, api_url: str):
    """
    Envía el diccionario metadata a la API.
    """
    print(" Enviando metadata a la API...")
    # print(f"Metadata: {metadata}", flush=True)
    try:
        response = requests.post(api_url, json=metadata)
        # print(f"Respuesta: {response}", flush=True)
        if response.status_code == 200:
            print(" Metadata enviada con éxito.")
        else:
            print(f" Error al enviar metadata: {response.status_code} - {response.text}")
    except Exception as e:
        print(f" Error al conectar con la API: {e}")


def prepare_and_send_alert(
    alert_type: str,
    ruma_data: RumaInfo = None,
    context: AlertContext = None,
    api_url: str = None,
):
    """Construye el metadata de alerta y lo envía usando send_metadata()."""
    timestamp = datetime.now()
    # Calcular tiempo del video
    #video_time_seconds = context.frame_count / context.fps

    # Metadata de la alerta
    
    # Inicializar img_b64 como None
    img_b64 = None
    
    # Determinar si se debe capturar y enviar la imagen en base64
    # Alertas: variacion_rumas, interaccion_rumas, movimiento_zona
    if alert_type in ["variacion_rumas", "interaccion_rumas", "movimiento_zona", "nueva_ruma"]:
        if context.frame is not None:
            img_b64 = frame_to_base64(context.frame)
    
    # Si es movimiento_zona y no hay coords válidas → forzar [0.0, 0.0]
    
    if alert_type == "movimiento_zona":
        if ruma_data.percent is None:
            ruma_data.percent = 0.0
        if ruma_data.radius_homographic is None:
            ruma_data.radius_homographic = 0.0
        if ruma_data.centroid_homographic is None:
            ruma_data.centroid_homographic = [0.0, 0.0]

    metadata = {
        "camera": context.camera_sn,
        "enterprise": context.enterprise,
        "customer": "falcon",
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "id": ruma_data.id, 
        "percent": ruma_data.percent,
        "coords": list(ruma_data.centroid_homographic), #ruma_data.centroid,
        "radius": ruma_data.radius_homographic, #ruma_data.radius,
        #"centroid_homographic": ruma_data.centroid_homographic,
        #"radius_homographic": ruma_data.radius_homographic,
        #"frame": None,
        "image": img_b64,  # base64 para las alertas especificadas

        #"frame_number": context.frame_count,
        #"video_time_seconds": video_time_seconds,
    }
    #-------------------
    debug_metadata = {k: v for k, v in metadata.items() if k != "image"}
    if img_b64:
        saved_path = save_b64_image(img_b64, alert_type)
        print("Imagen guardada en:", saved_path)

    print("Metadata sin imagen:", debug_metadata)
    #---------------------
    #send_metadata(metadata, api_url)

  
