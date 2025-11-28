#!/bin/bash

# Script para exportar ambos modelos a TensorRT
# Ejecutar en Jetson Orin Nano

echo "=========================================="
echo "Exportando modelos a TensorRT"
echo "=========================================="
echo ""

# Modelo de detección
echo "1/2 Exportando modelo de detección..."
yolo mode=export model=models/model_detection.pt format=engine device=0 half=True imgsz=640

echo ""
echo "2/2 Exportando modelo de segmentación..."
yolo mode=export model=models/model_segmentation.pt format=engine device=0 half=True imgsz=640

echo ""
echo "=========================================="
echo "✅ Exportación completada"
echo "=========================================="
echo ""
echo "Archivos generados:"
echo "  - models/model_detection.engine"
echo "  - models/model_segmentation.engine"
echo ""
echo "Para usar TensorRT, actualiza process_video.py:"
echo "  model_det_path = 'models/model_detection.engine'"
echo "  model_seg_path = 'models/model_segmentation.engine'"
