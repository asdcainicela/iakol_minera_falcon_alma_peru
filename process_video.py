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
    parser.add_argument("--target-size", type=int, default=1024, 
                       help="Tamaño máximo para resize (default: 1024)")

    args = parser.parse_args()

    camera_number = args.camera_number
    target_size = args.target_size

    try:
        (input_video, _, polygons, camera_sn, _, transformer, use_rtsp,
         save_video, start_video, end_video, time_save_rtsp) = load_camera_config(
            camera_number, config_path="mkdocs.yml"
        )
        
        # Determinar tiempos según use_rtsp
        if use_rtsp:
            if save_video:
                start_time_sec = 0
                end_time_sec = time_save_rtsp
                print(f"[INFO] Stream RTSP con grabación activa: {time_save_rtsp}s")
            else:
                start_time_sec = 0
                end_time_sec = float('inf')
                print("[INFO] Stream RTSP sin grabación: procesamiento continuo")
        else:
            start_time_sec = start_video
            end_time_sec = end_video
            print(f"[INFO] Video local: procesando desde {start_time_sec}s hasta {end_time_sec}s")
        
        # Generar ruta de salida
        camera_sn_clean = camera_sn.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_')
        output_video = generar_output_video(input_video, camera_sn=camera_sn_clean)
        
        print(f"Procesando video: {input_video}")
        print(f"Guardar video: {'Sí' if save_video else 'No'}")
        print(f"Tamaño objetivo para resize: {target_size}px")
        if save_video:
            print(f"Salida: {output_video}")
            
    except ValueError as e:
        print(f"[Error] {e}")
        return

    # MODELOS .engine
    model_det_path = 'models/model_detection.engine'
    model_seg_path = 'models/model_segmentation.engine'
    
    # Si no existen los .engine, intentar con .pt
    if not os.path.exists(model_det_path):
        print(f"[WARN] No se encontró {model_det_path}, intentando con .pt")
        model_det_path = 'models/model_detection.pt'
    
    if not os.path.exists(model_seg_path):
        print(f"[WARN] No se encontró {model_seg_path}, intentando con .pt")
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
        save_video=save_video,
        target_size=target_size  # NUEVO parámetro
    )

if __name__ == "__main__":
    main()