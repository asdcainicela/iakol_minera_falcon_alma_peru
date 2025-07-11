import os
from datetime import datetime
from pathlib import Path

def generar_ruta_salida(video_path):
    """
    Genera una ruta de salida para un video con el formato:
    misma carpeta que el video original, en una subcarpeta 'output/{nombre_input}/',
    y nombre: output_{nombre_input}_{fecha_hora}.mp4
    """
    folder, filename = os.path.split(video_path)
    name, ext = os.path.splitext(filename)
    fecha_hora = datetime.now().strftime('%Y%m%d_%H%M')
    output_filename = f'output_{name}_{fecha_hora}.mp4'

    # Carpeta de salida: misma carpeta + subcarpeta 'output/{nombre_input}'
    output_folder = os.path.join(folder, 'output', name)

    # Crear carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, output_filename)
    return output_path

def setup_alerts_folder():
        """Crea la estructura de carpetas para alertas"""
        alerts_base_path = Path("alerts_save")
        alerts_base_path.mkdir(exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        today_alerts_path = alerts_base_path / today
        today_alerts_path.mkdir(exist_ok=True)

if __name__ == "__main__":
    video_path = 'videos/video_cam2.mp4'
    output_path = generar_ruta_salida(video_path)
    print("Ruta generada:", output_path)
