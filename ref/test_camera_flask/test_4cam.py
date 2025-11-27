import os, time, cv2
from flask import Flask, Response, render_template_string, request, jsonify

# Configuración
USER = "admin"
PASS = "Facil.12"
PORT = 554

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

app = Flask(__name__)

# Configuración de las 4 cámaras
CAMERAS = [
    {"name": "Cámara 1", "ip": "192.168.1.2", "enabled": True},
    {"name": "Cámara 2", "ip": "192.168.1.3", "enabled": True},
    {"name": "Cámara 3", "ip": "192.168.1.47", "enabled": True},
    {"name": "Cámara 4", "ip": "192.168.1.64", "enabled": True},
]

def get_camera_url(camera_ip, sub=False):
    stream = "2" if sub else "1"
    return f"rtsp://{USER}:{PASS}@{camera_ip}:{PORT}/Streaming/Channels/{stream}01"

# Generador de frames para cada cámara
def generate_frames(camera_id):
    camera = CAMERAS[camera_id]
    url = get_camera_url(camera["ip"], sub=False)
    cap = None

    print(f"[Cam {camera_id + 1}] Conectando a: {camera['ip']}")

    while True:
        try:
            # Verifica si la cámara está habilitada
            if not CAMERAS[camera_id]["enabled"]:
                time.sleep(0.5)
                continue

            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    time.sleep(2)
                    continue
                print(f"[Cam {camera_id + 1}] ✓ Conectada")

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                time.sleep(1)
                continue

            # Redimensiona para mejor performance
            frame = cv2.resize(frame, (640, 480))

            # Codifica como JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            if cap:
                cap.release()
                cap = None
            time.sleep(2)

# API para toggle de cámaras
@app.route('/toggle/<int:camera_id>')
def toggle_camera(camera_id):
    if 0 <= camera_id < len(CAMERAS):
        CAMERAS[camera_id]["enabled"] = not CAMERAS[camera_id]["enabled"]
        status = "encendida" if CAMERAS[camera_id]["enabled"] else "apagada"
        print(f"[Cam {camera_id + 1}] {status}")
        return jsonify({"success": True, "enabled": CAMERAS[camera_id]["enabled"]})
    return jsonify({"success": False}), 404

# Rutas para cada cámara
@app.route('/video/<int:camera_id>')
def video_feed(camera_id):
    if camera_id < 0 or camera_id >= len(CAMERAS):
        return "Cámara no válida", 404
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Página principal
@app.route('/')
def index():
    cameras_html = ""
    for i, cam in enumerate(CAMERAS):
        enabled_class = "enabled" if cam["enabled"] else "disabled"
        cameras_html += f'''
        <div class="camera-box {enabled_class}" id="cam-{i}">
            <div class="camera-header">
                <div class="camera-label">{cam['name']}</div>
                <label class="switch">
                    <input type="checkbox" {"checked" if cam["enabled"] else ""} onchange="toggleCamera({i})">
                    <span class="slider"></span>
                </label>
            </div>
            <div class="camera-ip">📡 {cam['ip']}</div>
            <img src="/video/{i}" alt="{cam['name']}" class="camera-stream">
            <div class="offline-overlay">Cámara Desactivada</div>
        </div>
        '''

    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cámaras Hikvision</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                background: #1a1a1a;
                font-family: Arial, sans-serif;
                overflow: hidden;
            }
            .credentials {
                background: rgba(0, 0, 0, 0.9);
                color: #fbbf24;
                padding: 8px 20px;
                text-align: center;
                font-size: 12px;
                font-family: monospace;
                border-bottom: 2px solid #667eea;
            }
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 10px;
                padding: 10px;
                height: calc(100vh - 45px);
            }
            .camera-box {
                background: #2d2d2d;
                border-radius: 8px;
                overflow: hidden;
                position: relative;
                border: 2px solid #3d3d3d;
                transition: all 0.3s ease;
            }
            .camera-box.disabled {
                opacity: 0.5;
            }
            .camera-box.disabled .camera-stream {
                filter: grayscale(100%) blur(3px);
            }
            .camera-stream {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
                background: #000;
            }
            .camera-header {
                position: absolute;
                top: 10px;
                left: 10px;
                right: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                z-index: 10;
            }
            .camera-label {
                background: rgba(102, 126, 234, 0.95);
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .camera-ip {
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.8);
                color: #4ade80;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
                font-family: monospace;
                z-index: 10;
            }
            .offline-overlay {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(239, 68, 68, 0.9);
                color: white;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                display: none;
                z-index: 9;
            }
            .camera-box.disabled .offline-overlay {
                display: block;
            }

            /* Switch Toggle */
            .switch {
                position: relative;
                display: inline-block;
                width: 50px;
                height: 26px;
            }
            .switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ef4444;
                transition: .3s;
                border-radius: 26px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .slider:before {
                position: absolute;
                content: "";
                height: 18px;
                width: 18px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .3s;
                border-radius: 50%;
            }
            input:checked + .slider {
                background-color: #4ade80;
            }
            input:checked + .slider:before {
                transform: translateX(24px);
            }
        </style>
    </head>
    <body>
        <div class="credentials">
            🔐 Usuario: ''' + USER + ''' | Contraseña: ''' + PASS + '''
        </div>
        <div class="grid-container">
            ''' + cameras_html + '''
        </div>

        <script>
            function toggleCamera(camId) {
                fetch('/toggle/' + camId)
                    .then(response => response.json())
                    .then(data => {
                        const camBox = document.getElementById('cam-' + camId);
                        if (data.enabled) {
                            camBox.classList.remove('disabled');
                            camBox.classList.add('enabled');
                        } else {
                            camBox.classList.add('disabled');
                            camBox.classList.remove('enabled');
                        }
                    });
            }
        </script>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    print("=" * 70)
    print("🎥 Dashboard de 4 Cámaras Hikvision")
    print("=" * 70)
    for i, cam in enumerate(CAMERAS, 1):
        print(f"📹 Cámara {i}: {cam['name']} - IP: {cam['ip']}")
    print("=" * 70)
    print(f"🌐 Abre en tu navegador: http://192.168.1.196:5000")
    print("=" * 70)
    print("Presiona Ctrl+C para detener")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
