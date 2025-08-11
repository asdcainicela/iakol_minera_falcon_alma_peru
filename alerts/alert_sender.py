import requests
from datetime import datetime

from alerts.alert_info import RumaInfo, AlertContext

def send_metadata(metadata: dict, api_url: str):
    """
    Envía el diccionario metadata a la API.
    """
    print(" Enviando metadata a la API...")
    try:
        response = requests.post(api_url, json=metadata)
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
    metadata = {
        "cameraSN": context.camera_sn,
        "enterprise": context.enterprise,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "id": ruma_data.id, 
        "percent": ruma_data.percent,
        "coords": ruma_data.centroid_homographic, #ruma_data.centroid,
        "radius": ruma_data.radius_homographic, #ruma_data.radius,
        #"centroid_homographic": ruma_data.centroid_homographic,
        #"radius_homographic": ruma_data.radius_homographic,
        "frame": None,
        #"frame_number": context.frame_count,
        #"video_time_seconds": video_time_seconds,
    }

    send_metadata(metadata, api_url)

  
