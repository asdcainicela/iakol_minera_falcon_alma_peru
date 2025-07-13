import requests
from datetime import datetime

def send_metadata(metadata: dict, api_url: str):
    """
    Envía el diccionario metadata a la API.
    """
    print("🚀 Enviando metadata a la API...")
    try:
        response = requests.post(api_url, json=metadata)
        if response.status_code == 200:
            print(" Metadata enviada con éxito.")
        else:
            print(f" Error al enviar metadata: {response.status_code} - {response.text}")
    except Exception as e:
        print(f" Error al conectar con la API: {e}")

def prepare_and_send_alert(alert_type, frame_count, fps, camera_sn, enterprise, api_url):
    """
    Construye el metadata de alerta y lo envía usando send_metadata().
    """
    timestamp = datetime.now()
    metadata = {
        "cameraSN": camera_sn,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "video_time_seconds": frame_count / fps,
        "frame_number": frame_count,
        "enterprise": enterprise
    }

    send_metadata(metadata, api_url)
