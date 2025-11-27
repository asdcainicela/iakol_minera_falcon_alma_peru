#---
# name: l4t-ml
# group: ml
# depends: [pytorch, torchvision, torchaudio, tensorflow2, opencv, pycuda, cupy, onnxruntime, numba, gstreamer, jupyterlab]
# test: [test_pandas.py, test_scipy.py, test_sklearn.py]
#---
# Imagen correcta para JetPack 6.1 (L4T R36.4.x)
FROM nvcr.io/nvidia/l4t-ml:r36.4.0

# Directorio de trabajo
WORKDIR /unidas_system/files

# Herramientas de video y codecs
RUN apt-get update && apt-get install -y \
    v4l-utils \
    ffmpeg \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly

# Zona horaria
RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime && \
    dpkg-reconfigure -fnoninteractive tzdata

# Actualizar pip
RUN python3 -m pip install --upgrade pip

# Copiar requirements
COPY requirements_container.txt .

# Instalar paquetes seguros (JetPack 6)
RUN pip3 install --no-cache-dir pandas
RUN pip3 install --no-cache-dir scikit-learn

# Instalar requerimientos del proyecto
RUN pip3 install --no-cache-dir --default-timeout=1000 -r requirements_container.txt

# ncnn (sin dependencias de opencv)
RUN pip3 install ncnn --no-deps

# YOLOv8 ONNX + sin opencv
RUN pip3 install ultralytics==8.3.63 --no-deps

# Fix TensorRT numpy change (JetPack 6 usa numpy >=1.24)
RUN sed -i "s|bool: np.bool,|bool: np.bool_,|g" /usr/lib/python3.10/dist-packages/tensorrt/__init__.py

# Dependencias del proyecto
RUN pip3 install imutils --no-deps
RUN pip3 install logging-reset
RUN pip3 install supervision==0.18.0 --no-deps

# --- torch2trt (compatible JetPack 6) ---
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /unidas_system/files/torch2trt
RUN python3 setup.py install --user

# Volvemos al directorio principal
WORKDIR /app/alma/
