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

    args = parser.parse_args()

    camera_number = args.camera_number

    try:
        (input_video, _, polygons, camera_sn, _, transformer, use_rtsp,
         save_video, start_video, end_video, time_save_rtsp) = load_camera_config(
            camera_number, config_path="mkdocs.yml"
        )
        
        # Determinar tiempos según use_rtsp
        if use_rtsp:
            # Para RTSP
            if save_video:
                # Si se guarda video, usar time_save_rtsp del YAML
                start_time_sec = 0
                end_time_sec = time_save_rtsp
                print(f"[INFO] Stream RTSP con grabación activa: {time_save_rtsp}s")
            else:
                # Si NO se guarda video, procesar indefinidamente
                start_time_sec = 0
                end_time_sec = float('inf')
                print("[INFO] Stream RTSP sin grabación: procesamiento continuo")
        else:
            # Para archivos MP4, usar valores del YAML
            start_time_sec = start_video
            end_time_sec = end_video
            print(f"[INFO] Video local: procesando desde {start_time_sec}s hasta {end_time_sec}s")
        
        # Generar ruta de salida
        # Ejemplo sn DS-2SE3C204MWG-E/1220240711AAWRFH2517200 -> DS-2SE3C204MWG_E_1220240711AAWRFH2517200
        camera_sn_clean = camera_sn.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_')
        output_video = generar_output_video(input_video, camera_sn=camera_sn_clean)
        
        print(f"Procesando video: {input_video}")
        print(f"Guardar video: {'Sí' if save_video else 'No'}")
        if save_video:
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
        transformer=transformer,
        use_rtsp=use_rtsp,
        save_video=save_video
    )

if __name__ == "__main__":
    main()