from alerts.alert_sender import prepare_and_send_alert
from alerts.alert_storage import save_alert_local, RumaInfo, AlertContext
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
    detection_zone: Optional[List[Tuple[int, int]]] = None
) -> bool:
    """
    Orquesta el envío y guardado de una alerta.
    Retorna True si todo se ejecuta correctamente,
    False si hay errores.
    """
    try:
        if send:
            prepare_and_send_alert(
                alert_type=alert_type,
                frame_count=frame_count,
                fps=fps,
                camera_sn=camera_sn,
                enterprise=enterprise,
                api_url=api_url
            )

        if save:
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

            save_alert_local(
                alert_type=alert_type,
                ruma_data=ruma_data,
                context=context
            )

        return True

    except Exception as e:
        print(f"❌ Error en save_alert: {e}")
        return False
