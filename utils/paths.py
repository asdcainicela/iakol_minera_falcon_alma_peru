from pathlib import Path
from datetime import datetime

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

def generar_output_video(video_path: str) -> Path:
    """
    Genera una ruta de salida para un video procesado.

    Args:
        video_path (str): Ruta al video de entrada.

    Returns:
        Path: Ruta completa para guardar el video de salida.
    """
    video = Path(video_path)
    name = video.stem
    fecha_hora = datetime.now().strftime('%Y%m%d_%H%M')
    output_filename = f"output_{name}_{fecha_hora}.mp4"

    output_folder = video.parent / "output" / name
    output_folder.mkdir(parents=True, exist_ok=True)

    return output_folder / output_filename

if __name__ == "__main__":
    # Crear carpeta de alertas
    alerts_today = setup_alerts_folder()
    print(f"Carpeta de alertas creada en: {alerts_today}")

    # Ruta de salida para un video
    video_path = 'videos/video_cam2.mp4'
    output_path = generar_output_video(video_path)
    print(f"Ruta generada para guardar el video procesado: {output_path}")
