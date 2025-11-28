import os
import base64
import cv2
from datetime import datetime

def frame_to_base64(frame):
    """
    Convierte un frame de OpenCV a base64 JPEG comprimido.
    
    Args:
        frame: Frame de OpenCV en formato BGR
        
    Returns:
        str: String base64 del frame comprimido
    """
    # Resize a 320x180 (rectangular 16:9)
    small = cv2.resize(frame, (320, 180))
    
    # JPEG calidad 70 (muy liviano)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    ok, buffer = cv2.imencode(".jpg", small, encode_param)
    
    # Convertir a Base64
    jpg_b64 = base64.b64encode(buffer).decode("utf-8")
    
    return jpg_b64

def save_b64_image(b64_str: str, alert_type: str, folder="img_alert"):
    if b64_str is None:
        return None

    os.makedirs(folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{ts}.jpg"
    path = os.path.join(folder, filename)

    img_bytes = base64.b64decode(b64_str)

    with open(path, "wb") as f:
        f.write(img_bytes)

    return path
