"""
utils/export_to_tensorrt.py

Script para exportar modelos YOLO a TensorRT para Jetson Orin Nano
"""

from ultralytics import YOLO
import argparse


def export_model(model_path: str, imgsz: int = 640, half: bool = True):
    """
    Exporta un modelo YOLO a formato TensorRT (.engine)
    
    Args:
        model_path: Ruta al modelo .pt
        imgsz: Tamaño de imagen para inferencia
        half: Si True, usa FP16 (más rápido en Jetson)
    """
    print(f"\n{'='*70}")
    print(f"Exportando {model_path} a TensorRT")
    print(f"{'='*70}\n")
    
    model = YOLO(model_path)
    
    # Exportar a TensorRT
    success = model.export(
        format='engine',
        device=0,           # GPU 0
        half=half,          # FP16 para Jetson
        imgsz=imgsz,        # Tamaño de entrada
        workspace=4,        # 4GB workspace (ajustar según RAM disponible)
        verbose=True
    )
    
    if success:
        engine_path = model_path.replace('.pt', '.engine')
        print(f"\n✅ Modelo exportado exitosamente a: {engine_path}")
        print(f"   - Formato: TensorRT FP16")
        print(f"   - Input size: {imgsz}x{imgsz}")
        return engine_path
    else:
        print(f"\n❌ Error exportando modelo")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportar modelos YOLO a TensorRT")
    parser.add_argument("model_path", type=str, help="Ruta al modelo .pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--no-half", action="store_true", help="Desactivar FP16")
    
    args = parser.parse_args()
    
    export_model(args.model_path, imgsz=args.imgsz, half=not args.no_half)
