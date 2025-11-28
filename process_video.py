import os
import cv2
import time
import json
import base64
import torch
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
import argparse

# Librerías propias
from utils.paths import generar_output_video
from libs.config_loader import load_camera_config
from libs.main_functions import process_video


def main():
    parser = argparse.ArgumentParser(description="Procesamiento de video desde cámara.")
    parser.add_argument("camera_number", type=int, help="Número de cámara definido en el archivo mkdocs.yml")
    parser.add_argument("--start", type=float, default=0, help="Segundo inicial del video (default: 0)")
    parser.add_argument("--end", type=float, default=12, help="Segundo final del video (default: 12)")

    args = parser.parse_args()

    camera_number = args.camera_number
    start_time_sec = args.start
    end_time_sec = args.end

    try:
        input_video, _, polygons, camera_sn, _, transformer = load_camera_config(
            camera_number, config_path="mkdocs.yml"
        )
        # Generar ruta de salida pasando también el camera_sn
        output_video = generar_output_video(input_video, camera_sn=camera_sn)
        print(f"Procesando video: {input_video}")
        print(f"Salida: {output_video}")
    except ValueError as e:
        print(f"[Error] {e}")
        return

    model_det_path = 'models/model_detection.pt'
    model_seg_path = 'models/model_segmentation.pt'
    api_url = "https://api.ia-kol.com/api/Alert/create-alert-va"

    process_video(
        video_path=input_video,
        output_path=output_video,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        model_det_path=model_det_path,
        model_seg_path=model_seg_path,
        detection_zone=polygons,
        camera_number=camera_number,
        camera_sn=camera_sn,
        api_url=api_url,
        transformer=transformer
    )

    print("\n=== Procesamiento finalizado correctamente ===")


if __name__ == "__main__":
    main()