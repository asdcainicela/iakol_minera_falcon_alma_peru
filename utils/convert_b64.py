import cv2
import base64

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