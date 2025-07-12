# alerts/alert_sender.py
import requests

def send_metadata(metadata: dict, api_url: str):
    print("Enviando metadata a la API...")
    try:
        response = requests.post(api_url, json=metadata)
        if response.status_code == 200:
            print("Metadata enviada con éxito.")
        else:
            print(f"Error al enviar metadata: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error al conectar con la API: {e}")
