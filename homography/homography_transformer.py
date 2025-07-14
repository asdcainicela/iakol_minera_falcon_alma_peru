import cv2
import numpy as np

class HomographyTransformer:
    def __init__(self, input_pts, output_pts):
        self.input_pts = np.array(input_pts, dtype=np.float32)
        self.output_pts = np.array(output_pts, dtype=np.float32)
        self.H, _ = cv2.findHomography(self.input_pts, self.output_pts)
        print("[INFO] Matriz de homografía H:\n", self.H)

    def transform_point(self, point):
        """Transforma un solo punto (x, y)"""
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return transformed[0][0]

    def transform_points(self, points):
        """Transforma múltiples puntos [(x1, y1), (x2, y2), ...]"""
        pts = np.array([points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed[0]


if __name__ == "__main__":
    # === Coordenadas de entrada (desde imagen original)
    input_pts = [
        [1889, 454],
        [1122, 256],
        [120, 1070],
        [1441, 631],
        [1893,780]
    ]

    # === Coordenadas destino (en vista deseada)
    output_pts = [
        [837, 72],
        [324, 73],
        [339, 692],
        [575, 363],
        [758,390]
    ]

    transformer = HomographyTransformer(input_pts, output_pts)

    # === Puntos bien dentro del polígono original
    test_points = [
        (1000, 500),
        (1400, 600),
        (700, 800),
        (900, 700)
    ]

    transformed = transformer.transform_points(test_points)

    print("[TEST] Puntos internos transformados:")
    for original, mapped in zip(test_points, transformed):
        print(f"  {original} → {mapped}")
