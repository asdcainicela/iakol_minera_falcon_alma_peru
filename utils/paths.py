from pathlib import Path
from datetime import datetime
from typing import Union
from urllib.parse import urlparse

def setup_alerts_folder(base_path="alerts_save") -> Path:
    """
    Crea la carpeta base para guardar alertas organizada por fecha.
    
    Returns:
        Path: Ruta de la carpeta del día actual.
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    today_folder = base / today
    today_folder.mkdir(exist_ok=True)

    return today_folder

def generar_output_video(video_path: str, camera_sn: str = None) -> Path:
    """
    Genera una ruta de salida para un video procesado.
    Maneja tanto archivos locales como URLs RTSP.

    Args:
        video_path (str): Ruta al video de entrada o URL RTSP.
        camera_sn (str): Número de serie de la cámara (opcional).

    Returns:
        Path: Ruta completa para guardar el video de salida.
    """
    fecha_hora = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Detectar si es una URL RTSP
    if video_path.startswith('rtsp://') or video_path.startswith('http://'):
        # Es un stream, usar el nombre de la cámara
        if camera_sn:
            name = camera_sn.replace('-', '_')
        else:
            # Extraer algún identificador de la URL
            parsed = urlparse(video_path)
            # Usar el hostname o una parte de la ruta
            name = parsed.hostname.replace('.', '_') if parsed.hostname else "stream"
        
        output_filename = f"output_{name}_{fecha_hora}.mp4"
        output_folder = Path("video") / "output" / name
    else:
        # Es un archivo local
        video = Path(video_path)
        name = video.stem
        output_filename = f"output_{name}_{fecha_hora}.mp4"
        output_folder = video.parent / "output" / name
    
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder / output_filename

def generar_folder_fecha(base_path: Union[str, Path], etiqueta: str = "local") -> Path:
    """
    Crea una carpeta dentro de `base_path` con la fecha actual y una etiqueta opcional.

    Por ejemplo: 'alerts_save/2025-07-13_local'

    Args:
        base_path (str | Path): Ruta base donde crear la carpeta.
        etiqueta (str): Sufijo adicional para distinguir carpetas (por defecto "local").

    Returns:
        Path: Ruta completa a la carpeta creada.
    """
    base = Path(base_path)
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"{date_str}_{etiqueta}"
    output_path = base / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

if __name__ == "__main__":
    # Crear carpeta de alertas
    alerts_today = setup_alerts_folder()
    print(f"Carpeta de alertas creada en: {alerts_today}")

    # Prueba con archivo local
    video_path_local = 'videos/video_cam2.mp4'
    output_path_local = generar_output_video(video_path_local)
    print(f"Ruta para video local: {output_path_local}")
    
    # Prueba con RTSP
    video_path_rtsp = 'rtsp://admin:Facil.12@192.168.0.3:554/cam/realmonitor?channel=1&subtype=0'
    output_path_rtsp = generar_output_video(video_path_rtsp, camera_sn="DS-7104NI-Q1-1")
    print(f"Ruta para RTSP: {output_path_rtsp}")