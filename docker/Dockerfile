#---
# name: l4t-ml
# group: ml
# depends: [pytorch, torchvision, torchaudio, tensorflow2, opencv, pycuda, cupy, onnxruntime, numba, gstreamer, jupyterlab]
# test: [test_pandas.py, test_scipy.py, test_sklearn.py]
#---
# Imagen oficial correcta para JetPack 6.1 (L4T R36.4.x)
FROM dustynv/l4t-ml:r36.4.0

# Directorio de trabajo UNIFICADO
WORKDIR /app/alma

# Herramientas de video y codecs
RUN apt-get update && apt-get install -y \
    v4l-utils \
    ffmpeg \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    && rm -rf /var/lib/apt/lists/*

# Zona horaria Lima
RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime && \
    dpkg-reconfigure -fnoninteractive tzdata

# PRIMERO: Limpiar configuración de pip de la imagen base
RUN rm -f /etc/pip.conf ~/.pip/pip.conf /root/.pip/pip.conf || true

# SEGUNDO: Actualizar pip usando PyPI oficial directamente
RUN python3 -m pip install --upgrade pip --index-url https://pypi.org/simple

# TERCERO: Configurar pip globalmente para usar solo PyPI oficial
RUN pip config set global.index-url https://pypi.org/simple && \
    pip config set global.extra-index-url ""

# CUARTO: Instalar paquetes grandes/problemáticos PRIMERO desde PyPI oficial
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple \
    scipy>=1.4.1 \
    pandas>=1.1.4 \
    scikit-learn \
    numpy \
    pillow \
    requests>=2.23.0 \
    pyyaml>=5.3.1

# Copiar requirements al directorio de trabajo
COPY requirements_container.txt .

# Instalar requerimientos del proyecto FORZANDO PyPI oficial
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple --default-timeout=1000 -r requirements_container.txt

# NCNN sin dependencias pesadas
RUN pip3 install ncnn --no-deps

# YOLOv8 sin opencv
RUN pip3 install ultralytics==8.3.63 --no-deps

# Fix TensorRT para numpy >= 1.24 (JetPack 6)
RUN sed -i "s|bool: np.bool,|bool: np.bool_,|g" /usr/lib/python3.10/dist-packages/tensorrt/__init__.py

# Dependencias del proyecto
RUN pip3 install imutils --no-deps \
    && pip3 install logging-reset \
    && pip3 install supervision==0.18.0 --no-deps

# torch2trt (compatible JetPack 6) - instalación temporal
RUN cd /tmp && \
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install --user && \
    cd / && \
    rm -rf /tmp/torch2trt

# Volver al directorio de trabajo final
WORKDIR /app/alma