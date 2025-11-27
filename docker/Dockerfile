#---
# name: l4t-ml
# group: ml
# depends: [pytorch, torchvision, torchaudio, tensorflow2, opencv, pycuda, cupy, onnxruntime, numba, gstreamer, jupyterlab]
# test: [test_pandas.py, test_scipy.py, test_sklearn.py]
#---
FROM dustynv/l4t-ml:r35.4.1

# bash
# SHELL ["/bin/bash", "-c"]

WORKDIR /unidas_system/files

# instala utilidades de vídeo y códecs
RUN apt-get update && \
    apt-get install -y \
      v4l-utils \
      ffmpeg \
      gstreamer1.0-libav \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly

COPY requirements_container.txt .
#COPY ./src/docker_tests/ocr_image_test.jpg .
#COPY ./src/docker_tests/test_paddleocr.py .

# create folder haarcascade
RUN mkdir -p /usr/local/lib/python3.8/dist-packages/cv2/python-3.8/data/
#RUN cp -a haarcascades/ /usr/local/lib/python3.8/dist-packages/cv2/python-3.8/data/

# change zone time
RUN ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime
RUN dpkg-reconfigure -fnoninteractive tzdata

# update pip
RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir --verbose scipy
RUN pip3 install --no-cache-dir --verbose scikit-learn
RUN pip3 install --no-cache-dir --verbose pandas

# install container requirements
#RUN pip3 install -r requirements_container.txt
RUN pip3 install --no-cache-dir --default-timeout=1000 -r requirements_container.txt
# yolov8 benchmark # to no install opencv
RUN pip3 install ncnn --no-deps
# install yolov8
RUN pip3 install ultralytics==8.3.63 --no-deps
# fix numpy dependecy
RUN sed -i "s|bool: np.bool,|bool: np.bool_,|g" /usr/lib/python3.8/dist-packages/tensorrt/__init__.py

# PaddlePaddle
# paddleocr dependencie
RUN pip3 install imgaug --no-deps
# paddleocr
RUN pip3 install paddleocr==2.7.0.3 --no-deps

# project dependencies
RUN pip3 install imutils --no-deps
RUN pip3 install logging-reset
RUN pip3 install supervision==0.18.0 --no-deps

## Face recognition dependencies
# install deepface
RUN pip3 install deepface --no-deps
# retinaface
RUN pip3 install retina-face>=0.0.1 --no-deps
# mtcnn
RUN pip3 install mtcnn>=0.1.0 --no-deps

# Yolo project weights
#RUN yolo mode=export model=yolov8s.pt format=engine device=0 half=True

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /unidas_system/files/torch2trt
RUN python3 setup.py install --user



