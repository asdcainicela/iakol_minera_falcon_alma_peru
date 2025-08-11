import csv
import os
from datetime import datetime
from alerts.alert_info import RumaInfo, AlertContext

def save_to_csv(metadata: dict, csv_file: str = "alerts_data.csv"):
    """
    Guarda el diccionario metadata en un archivo CSV.
    Si el archivo existe, agrega los datos sin borrar los anteriores.
    """
    print(f" Guardando metadata en CSV: {csv_file}")
    
    try:
        # Verificar si el archivo existe
        file_exists = os.path.exists(csv_file)
        
        # Abrir archivo en modo append si existe, sino en modo write
        mode = 'a' if file_exists else 'w'
        
        with open(csv_file, mode, newline='', encoding='utf-8') as file:
            fieldnames = metadata.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Escribir header solo si es un archivo nuevo
            if not file_exists:
                writer.writeheader()
            
            # Escribir la fila de datos
            writer.writerow(metadata)
            
        print(f" Metadata guardada con éxito en CSV.")
        
    except Exception as e:
        print(f" Error al guardar en CSV: {e}")

def prepare_and_save_alert(
    alert_type: str,
    ruma_data: RumaInfo = None,
    context: AlertContext = None,
    csv_file: str = "alerts_data.csv",
):
    """
    Construye el metadata de alerta y lo guarda en CSV usando save_to_csv().
    """
    timestamp = datetime.now()
    
    # Metadata de la alerta
    metadata = {
        "cameraSN": context.camera_sn,
        "enterprise": context.enterprise,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "id": ruma_data.id,
        "percent": ruma_data.percent,
        "coords": str(ruma_data.centroid_homographic),  # Convertir a string para CSV
        "radius": ruma_data.radius_homographic,
        "frame": None,
    }
    
    save_to_csv(metadata, csv_file)

# Función alternativa que permite alternar entre API y Excel
def prepare_and_send_alert_flexible(
    alert_type: str,
    ruma_data: RumaInfo = None,
    context: AlertContext = None,
    api_url: str = None,
    use_csv: bool = False,
    csv_file: str = "alerts_data.csv",
):
    """
    Función flexible que puede enviar a API o guardar en CSV según el parámetro use_csv.
    """
    timestamp = datetime.now()
    
    # Metadata de la alerta
    metadata = {
        "cameraSN": context.camera_sn,
        "enterprise": context.enterprise,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "id": ruma_data.id,
        "percent": ruma_data.percent,
        "coords": str(ruma_data.centroid_homographic) if use_csv else ruma_data.centroid_homographic,
        "radius": ruma_data.radius_homographic,
        "frame": None,
    }
    
    if use_csv:
        save_to_csv(metadata, csv_file)
    else:
        # Aquí iría tu función original send_metadata
        send_metadata(metadata, api_url)

# Función original para referencia (sin cambios)
def send_metadata(metadata: dict, api_url: str):
    """
    Envía el diccionario metadata a la API.
    """
    import requests
    print(" Enviando metadata a la API...")
    try:
        response = requests.post(api_url, json=metadata)
        if response.status_code == 200:
            print(" Metadata enviada con éxito.")
        else:
            print(f" Error al enviar metadata: {response.status_code} - {response.text}")
    except Exception as e:
        print(f" Error al conectar con la API: {e}")