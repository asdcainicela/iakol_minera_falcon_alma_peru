from alerts.alert_sender import prepare_and_send_alert
from alerts.alert_storage import save_alert_local
from alerts.alert_info import RumaInfo, AlertContext
import numpy as np
from typing import Optional, Tuple, List

def save_alert(
    alert_type: str,
    ruma_data: RumaInfo,
    frame: np.ndarray,
    frame_count: int,
    fps: float,
    camera_sn: str,
    enterprise: str,
    api_url: str,
    send: bool = True,
    save: bool = True,
    ruma_summary: Optional[dict] = None,
    frame_shape: Optional[Tuple[int, int]] = None,
    detection_zone: Optional[List[Tuple[int, int]]] = None,
    # Nuevos parámetros para CSV
    save_csv: bool = True,  # Por defecto True para guardar CSV
    csv_file: str = "alerts_data.csv"
) -> bool:
    """
    Orquesta el envío y guardado de una alerta.
    
    Args:
        save_csv: Si True, guarda también en CSV
        csv_file: Nombre del archivo CSV
    
    Retorna True si todo se ejecuta correctamente, False si hay errores.
    """
    try:
        context = AlertContext(
            frame=frame,
            frame_count=frame_count,
            fps=fps,
            camera_sn=camera_sn,
            enterprise=enterprise,
            ruma_summary=ruma_summary,
            frame_shape=frame_shape,
            detection_zone=detection_zone
        )

        if send:
            prepare_and_send_alert(
                alert_type=alert_type,
                ruma_data=ruma_data,
                context=context,
                api_url=api_url
            )
            
        if save:
            save_alert_local(
                alert_type=alert_type,
                ruma_data=ruma_data,
                context=context,
                save_csv=save_csv,  # Nuevo parámetro
                csv_file=csv_file   # Nuevo parámetro
            )

        return True

    except Exception as e:
        print(f"Error en save_alert: {e}")
        return False