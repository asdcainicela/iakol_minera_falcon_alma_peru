# alerts/alert_manager.py
from datetime import datetime
from alerts.alert_sender import send_metadata
from alerts.alert_storage import save_alert_local
from utils.paths import setup_alerts_folder

def save_alert(alert_type, frame, frame_count, fps, camera_sn, enterprise, api_url,
               send=True, save=True, ruma_summary=None, frame_shape=None, detection_zone = None ):
    timestamp = datetime.now() 
    metadata = {
        "cameraSN": camera_sn,
        "alert_type": alert_type,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "video_time_seconds": frame_count / fps,
        "frame_number": frame_count,
        "enterprise": enterprise
    }

    if send:
        send_metadata(metadata, api_url)
    
    if save:
      # Crear carpeta de alertas
      setup_alerts_folder()
      # crear logica
      save_alert_local(
          alert_type=alert_type,
          frame=frame,
          frame_count=frame_count,
          fps=fps,
          camera_sn=camera_sn,
          enterprise=enterprise,
          ruma_summary=ruma_summary,
          frame_shape=frame_shape,
          detection_zone = detection_zone
      )

 
