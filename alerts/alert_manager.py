from alerts.alert_sender import prepare_and_send_alert
from alerts.alert_storage import save_alert_local

def save_alert(alert_type, frame, frame_count, fps, camera_sn, enterprise, api_url,
               send=True, save=True, ruma_summary=None, frame_shape=None, detection_zone=None) -> bool:
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
            save_alert_local(
                alert_type=alert_type,
                frame=frame,
                frame_count=frame_count,
                fps=fps,
                camera_sn=camera_sn,
                enterprise=enterprise,
                ruma_summary=ruma_summary,
                frame_shape=frame_shape,
                detection_zone=detection_zone
            )

        return True

    except Exception as e:
        print(f" Error en save_alert: {e}")
        return False
