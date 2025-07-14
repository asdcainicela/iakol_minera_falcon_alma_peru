import cv2
import numpy as np
import cv2
import numpy as np

class HomographyTransformer:
    def __init__(self, input_pts, output_pts):
        self.input_pts = np.array(input_pts, dtype=np.float32)
        self.output_pts = np.array(output_pts, dtype=np.float32)
        self.H, _ = cv2.findHomography(self.input_pts, self.output_pts)
        print("[INFO] Matriz de homografía H:\n", self.H)

    def transform_point(self, point):
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return transformed[0][0]

    def transform_points(self, points):
        pts = np.array([points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed[0]

    def transform_circle(self, center, radius):
        """
        Transforma un círculo definido por un centro y un radio.
        Devuelve el nuevo centro y radio proyectado.
        """
        point_on_circle = (center[0] + radius, center[1])
        new_center = self.transform_point(center)
        new_edge = self.transform_point(point_on_circle)
        new_radius = np.linalg.norm(new_edge - new_center)
        return new_center, new_radius


if __name__ == "__main__":
    input_pts = [
        [1889, 454],
        [1122, 256],
        [120, 1070],
        [1441, 631],
        [1893, 780]
    ]

    output_pts = [
        [837, 72],
        [324, 73],
        [339, 692],
        [575, 363],
        [758, 390]
    ]

    transformer = HomographyTransformer(input_pts, output_pts)

    # Círculo original
    centroide = (1000, 500)
    radio_original = 50

    centro_transformado, radio_transformado = transformer.transform_circle(centroide, radio_original)

    print(f"[INFO] Centro original: {centroide}")
    print(f"[INFO] Centro transformado: {centro_transformado}")
    print(f"[INFO] Radio original: {radio_original}")
    print(f"[INFO] Radio transformado: {radio_transformado}")
