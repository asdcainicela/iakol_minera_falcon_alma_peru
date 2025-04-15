import cv2

url = "rtsp://admin:Perfumeriasunidas2!@192.168.0.102:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ No se pudo conectar al stream.")
else:
    print("✅ Conectado. Presioná Q para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('RTSP Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
