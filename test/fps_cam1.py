import cv2
import time

# URL de la cámara 1 - canal 101 (mainstream)
rtsp_url = "rtsp://admin:Facil.12@192.168.1.3:554/Streaming/Channels/101"

print("Conectando a la cámara...")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR: No se pudo conectar")
    exit()

print("Conectado!")

# Leer propiedades
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_opencv = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolución: {width}x{height}")
print(f"FPS reportado por OpenCV: {fps_opencv}")
print("\nMidiendo FPS real por 10 segundos...\n")

# Medir FPS real
frame_count = 0
start_time = time.time()

while time.time() - start_time < 100:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        
        # Imprimir cada segundo
        elapsed = time.time() - start_time
        if frame_count % 10 == 0:  # Cada 10 frames
            fps_actual = frame_count / elapsed
            print(f"{elapsed:.1f}s - Frames: {frame_count} - FPS: {fps_actual:.2f}")

end_time = time.time()
total_time = end_time - start_time
fps_real = frame_count / total_time

cap.release()

print("\n" + "="*50)
print(f"RESULTADO:")
print(f"Tiempo: {total_time:.2f}s")
print(f"Frames capturados: {frame_count}")
print(f"FPS REAL: {fps_real:.2f}")
print("="*50)